"""
Small grasp-side helper for reBotArm visual grasping.

The SDK remains responsible for arm connection, mode switching, Cartesian
planning, gravity compensation, and the control loop. This module only covers
the pieces that the vision stack needs in addition to the SDK:

    - resolve the local SDK path
    - read the SDK YAML to identify the selected arm type
    - open / force-grasp / release the gripper
    - read the current TCP pose
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


_CAMERAWS_ROOT = Path(__file__).resolve().parents[2]
_REBOT_REPO_NAME = "reBotArm_control_py"
_DEFAULT_REBOT_REPO = _CAMERAWS_ROOT / "sdk" / _REBOT_REPO_NAME

GRIPPER_MAX_DISTANCE_M = 0.09


@dataclass(frozen=True)
class SelectedArmConfig:
    arm_type: str
    controller_mode: str


def _is_rebot_repo_root(path: Path) -> bool:
    pkg = path / _REBOT_REPO_NAME
    return (
        path.is_dir()
        and (pkg / "actuator" / "rebotarm.py").is_file()
        and (path / "config" / "rebotarm.yaml").is_file()
    )


def find_rebot_repo_root(hint: Optional[str] = None) -> Path:
    repo = Path(hint).expanduser() if hint else _DEFAULT_REBOT_REPO
    if not repo.is_absolute():
        repo = (_CAMERAWS_ROOT / repo).resolve()
    else:
        repo = repo.resolve()
    if _is_rebot_repo_root(repo):
        return repo
    raise FileNotFoundError(f"找不到 reBotArm_control_py 仓库: {repo}")


def ensure_rebot_sdk_in_syspath(hint: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(hint)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 格式应为 YAML 字典")
    return data


def selected_hardware_yaml(repo_root: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(repo_root)
    config_dir = repo / "config"
    global_cfg = _read_yaml(config_dir / "rebotarm.yaml")
    hw_yaml = global_cfg.get("hardware_yaml")
    if not hw_yaml:
        raise ValueError(f"{config_dir / 'rebotarm.yaml'} 缺少 hardware_yaml")

    hw_path = Path(str(hw_yaml))
    if not hw_path.is_absolute():
        hw_path = config_dir / hw_path
    hw_path = hw_path.resolve()
    if not hw_path.is_file():
        raise FileNotFoundError(f"找不到硬件配置: {hw_path}")
    return hw_path


def selected_arm_config(repo_root: Optional[str] = None) -> SelectedArmConfig:
    """Return the selected arm type and matching SDK controller mode."""
    hw_path = selected_hardware_yaml(repo_root)
    stem = hw_path.stem.lower()
    if stem.endswith("_dm") or stem == "dm":
        return SelectedArmConfig(arm_type="dm", controller_mode="posvel")
    if stem.endswith("_rs") or stem == "rs":
        return SelectedArmConfig(arm_type="rs", controller_mode="mit")
    raise ValueError(f"无法从硬件配置判断机械臂类型: {hw_path}")


class GraspDriver:
    MAX_DISTANCE_M = GRIPPER_MAX_DISTANCE_M

    def __init__(
        self,
        arm: Any,
        controller: Any,
        gripper_config: Optional[dict] = None,
        repo_root: Optional[str] = None,
    ) -> None:
        self._arm = arm
        self._controller = controller
        self._arm_group = arm.groups.get("arm")
        self._gripper_group = arm.groups.get("gripper")
        if self._arm_group is None:
            raise ValueError("硬件配置缺少 groups.arm")
        if self._gripper_group is None or not arm.has_gripper:
            raise ValueError("硬件配置缺少 groups.gripper")

        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model

        self._compute_fk = compute_fk
        self._pad_q_for_model = pad_q_for_model
        self._model = load_robot_model()
        self._n = self._arm_group.num_joints

        selected = selected_arm_config(repo_root)
        defaults = {
            "dm": {"angle_open": -5.0, "tau_max": 1.5, "close_torque": 1.0, "default_force": 0.30},
            "rs": {"angle_open": 5.0, "tau_max": 1.5, "close_torque": -1.0, "default_force": -0.30},
        }[selected.arm_type]
        gcfg = {**defaults, **((gripper_config or {}).get(selected.arm_type) or {})}
        self._angle_open = float(gcfg["angle_open"])
        self._tau_max = abs(float(gcfg["tau_max"]))
        self._close_torque = float(gcfg["close_torque"])
        self._default_force = float(gcfg["default_force"])
        self._open_sign = 1.0 if self._angle_open >= 0.0 else -1.0
        self._open_soft_limit = 0.98 * self._angle_open
        self._hard_stop_angle = self._open_sign * 0.05
        self._max_dist_m = GRIPPER_MAX_DISTANCE_M
        self._arrive_tol = 0.12
        self._kp_move = 5.0
        self._kd_move = 1.0
        self._kd_close = 0.5
        self._stall_vel = 0.05
        self._startup_dist = 0.30

    def _send_gripper_mit(
        self,
        pos: float,
        vel: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        tau: float = 0.0,
    ) -> None:
        lo = min(self._open_soft_limit, 0.0)
        hi = max(self._open_soft_limit, 0.0)
        pos_cmd = float(np.clip(pos, lo, hi))
        tau_cmd = float(np.clip(tau, -self._tau_max, self._tau_max))
        self._gripper_group.send_mit(
            np.array([pos_cmd], dtype=np.float64),
            vel=np.array([vel], dtype=np.float64),
            kp=np.array([kp], dtype=np.float64),
            kd=np.array([kd], dtype=np.float64),
            tau=np.array([tau_cmd], dtype=np.float64),
        )

    def get_gripper_state(self) -> tuple[float, float, float]:
        self._gripper_group._request_feedback()
        jcfgs = getattr(self._gripper_group, "_jcfgs", [])
        if not jcfgs:
            raise RuntimeError("groups.gripper 未配置关节")
        mot = self._gripper_group._mm[jcfgs[0].name]
        st = mot.get_state()
        if st is None:
            raise RuntimeError("夹爪反馈未就绪")
        return (float(st.pos), float(st.vel), float(st.torq))

    def open_gripper(self, distance_m: float = GRIPPER_MAX_DISTANCE_M, timeout: float = 3.0) -> None:
        d = float(np.clip(distance_m, 0.0, self._max_dist_m))
        raw_target = (d / self._max_dist_m) * self._angle_open
        lo = min(self._open_soft_limit, 0.0)
        hi = max(self._open_soft_limit, 0.0)
        target = float(np.clip(raw_target, lo, hi))

        self._controller.set_gripper_target(target)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos, _, _ = self.get_gripper_state()
            self._send_gripper_mit(target, kp=self._kp_move, kd=self._kd_move)
            if abs(pos - target) < self._arrive_tol:
                return
            time.sleep(0.02)

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        start_pos, _, _ = self.get_gripper_state()
        sign = 1.0 if self._default_force >= 0.0 else -1.0
        hold_torque = sign * float(np.clip(abs(force if force is not None else self._default_force), 0.05, self._tau_max))
        self._controller.set_gripper_target(0.0)

        deadline = time.monotonic() + timeout
        contact_pos = start_pos
        while time.monotonic() < deadline:
            self._send_gripper_mit(0.0, kd=self._kd_close, tau=self._close_torque)
            pos, vel, _ = self.get_gripper_state()
            contact_pos = pos
            moved = abs(pos - start_pos) >= self._startup_dist
            at_hard_stop = self._open_sign * pos < self._open_sign * self._hard_stop_angle
            if moved and at_hard_stop:
                return False
            if moved and abs(vel) < self._stall_vel:
                self._controller.set_gripper_target(contact_pos)
                self._send_gripper_mit(contact_pos, kp=self._kp_move, kd=self._kd_move, tau=hold_torque)
                return True
            time.sleep(0.02)

        self._controller.set_gripper_target(contact_pos)
        self._send_gripper_mit(contact_pos, kp=self._kp_move, kd=self._kd_move)
        return False

    def release_gripper(self, timeout: float = 4.0) -> None:
        self.open_gripper(timeout=min(2.0, timeout))
        self._controller.set_gripper_target(0.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos, _, _ = self.get_gripper_state()
            self._send_gripper_mit(0.0, kp=self._kp_move, kd=self._kd_move)
            if abs(pos) < self._arrive_tol:
                return
            time.sleep(0.02)

    def get_tcp_pose(self) -> np.ndarray:
        q_arm = self._arm.get_state()[0][: self._n]
        q = self._pad_q_for_model(self._model, q_arm, self._n)
        pos, rot, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T
