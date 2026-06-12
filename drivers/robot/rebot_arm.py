"""
drivers.robot.rebot_arm
=================================
Lightweight adapter for the reBotArm_control_py grouped driver.

The visual grasping scripts keep using the original high-level interface:

    connect()              - enable the arm and start the control loop
    disconnect()           - disable and close the driver
    get_tcp_pose()         - read FK as a 4x4 T_gripper2base matrix
    move_to(x,y,z)         - IK + Cartesian trajectory to a TCP pose
    safe_home()            - return arm joints to zero

    init_gripper()         - validate the configured gripper group
    open_gripper(dist)     - open gripper and wait until idle
    close_gripper()        - torque-close, non-blocking
    grasp(force, timeout)  - close -> contact detect -> hold, blocking
    release_gripper()      - open and return gripper to zero
    get_gripper_state()    - read (pos, vel, torq)
    set_gripper_zero()     - set current gripper position as zero
    gripper_is_holding     - True while contact hold is active
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np



_CAMERAWS_ROOT = Path(__file__).resolve().parents[2]
_REBOT_REPO_NAME = "reBotArm_control_py"


def _is_rebot_repo_root(path: Path) -> bool:
    pkg = path / _REBOT_REPO_NAME
    return (
        path.is_dir()
        and (pkg / "actuator" / "rebotarm.py").is_file()
        and (path / "config" / "rebotarm.yaml").is_file()
    )


def find_rebot_repo_root(hint: Optional[str] = None) -> Path:
    if hint:
        hinted = Path(hint).expanduser()
        if not hinted.is_absolute():
            hinted = _CAMERAWS_ROOT / hinted
        hinted = hinted.resolve()
        if _is_rebot_repo_root(hinted):
            return hinted
        raise FileNotFoundError(f"找不到 reBotArm_control_py 仓库: {hinted}")

    repo = (_CAMERAWS_ROOT.parent / _REBOT_REPO_NAME).resolve()
    if _is_rebot_repo_root(repo):
        return repo
    raise FileNotFoundError(f"找不到 reBotArm_control_py 仓库: {repo}")


def ensure_rebot_sdk_in_syspath(hint: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(hint)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _first_joint_vendor(group) -> str:
    jcfgs = getattr(group, "_jcfgs", [])
    if not jcfgs:
        return ""
    return str(getattr(jcfgs[0], "vendor", "")).lower()


def _vendor_key(vendor: str) -> str:
    vendor = vendor.lower()
    if vendor == "robstride":
        return "rs"
    if vendor == "damiao":
        return "dm"
    return vendor


def _normalize_arm_control_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode == "pos_vel":
        mode = "posvel"
    if mode not in ("mit", "posvel"):
        raise ValueError("robot.control.*.arm_control_mode 必须为 mit 或 posvel")
    return mode


def _default_arm_control_mode(vendor: str) -> str:
    if vendor.lower() == "robstride":
        return "mit"
    return "posvel"


class _GS:
    IDLE    = 0
    OPENING = 1
    CLOSING = 2
    CONTACT = 3
    HOLDING = 4
    HOMING  = 5


class RebotArm:
    """Camera perception system <-> reBotArm adapter.

    Args:
        repo_root: reBotArm_control_py 仓库根目录；None 时自动查找同级目录。
        gripper_config: robot.gripper 配置字典，含 dm / rs 两组参数，按夹爪电机厂商自动选择
        control_config: robot.control 配置字典，含 dm / rs 两组 arm_control_mode 覆写
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        gripper_config: Optional[dict] = None,
        control_config: Optional[dict] = None,
    ) -> None:
        repo = ensure_rebot_sdk_in_syspath(repo_root)
        self._repo_root = repo

        from reBotArm_control_py.actuator import RebotArm as DriverRebotArm
        from reBotArm_control_py.kinematics import (
            IKSolverParams,
            compute_fk,
            get_end_effector_frame_id,
            load_robot_model,
            pad_q_for_model,
            pos_rot_to_se3,
        )
        from reBotArm_control_py.controllers import RebotArmEndPose
        from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik

        self._arm = DriverRebotArm()
        self._arm_group = self._arm.groups.get("arm")
        self._gripper_group = self._arm.groups.get("gripper")
        if self._arm_group is None:
            raise ValueError("硬件配置缺少 groups.arm")
        if not self._arm.has_gripper:
            raise ValueError(
                "硬件配置缺少 groups.gripper：抓取系统要求配置夹爪，"
                "请在 reBotArm_control_py/config 所选硬件 YAML 中启用 gripper 组"
            )
        self._n = self._arm_group.num_joints

        arm_vendor = _first_joint_vendor(self._arm_group)
        arm_type = _vendor_key(arm_vendor)
        control_for_type = (control_config or {}).get(arm_type) or {}
        control_override = _normalize_arm_control_mode(
            control_for_type.get("arm_control_mode")
            if isinstance(control_for_type, dict)
            else None
        )
        self._arm_control_mode = control_override or _default_arm_control_mode(arm_vendor)

        self._model = load_robot_model()
        self._data = self._model.createData()
        self._ee_frame_id = get_end_effector_frame_id(self._model)

        self._compute_fk = compute_fk
        self._pad_q_for_model = pad_q_for_model
        self._pos_rot_to_se3 = pos_rot_to_se3
        self._solve_ik = solve_ik
        self._ik_check_params = IKSolverParams(
            max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6,
        )

        self._endpose = RebotArmEndPose(self._arm, arm_control_mode=self._arm_control_mode)
        self._endpose._has_gripper = False

        self._connected = False
        self._control_active = False
        self._io_lock = threading.RLock()
        self._loop_err_t = 0.0

        self._g_state = _GS.IDLE
        self._g_lock = threading.Lock()
        self._g_pos = 0.0
        self._g_vel = 0.0
        self._g_torq = 0.0
        self._g_pos_start = 0.0
        self._g_q_contact = 0.0
        self._g_contact_elapsed = 0.0
        gripper_type = _vendor_key(_first_joint_vendor(self._gripper_group))
        defaults = {
            "dm": {"angle_open": -5.0, "tau_max": 1.5, "close_torque": 1.0, "default_force": 0.30},
            "rs": {"angle_open": 5.0, "tau_max": 1.5, "close_torque": -1.0, "default_force": -0.30},
        }.get(gripper_type, {"angle_open": -5.0, "tau_max": 1.5, "close_torque": 1.0, "default_force": 0.30})
        gcfg = {**defaults, **((gripper_config or {}).get(gripper_type) or {})}
        self._g_angle_open = float(gcfg["angle_open"])
        self._g_tau_max = abs(float(gcfg["tau_max"]))
        self._g_close_torque = float(gcfg["close_torque"])
        self._g_default_force = float(gcfg["default_force"])
        self._g_open_sign = 1.0 if self._g_angle_open >= 0.0 else -1.0
        self._g_open_soft_limit = 0.98 * self._g_angle_open
        self._g_hard_stop_angle = self._g_open_sign * 0.05
        self._g_max_dist_m = 0.09
        self._g_arrive_tol = 0.12
        self._g_kp_move = 5.0
        self._g_kd_move = 1.0
        self._g_open_rate = 4.0
        self._g_kd_close = 0.5
        self._g_stall_vel = 0.05
        self._g_startup_dist = 0.30
        self._g_kp_hold = 5.0
        self._g_kd_hold = 1.0
        self._g_open_q_des = self._g_open_soft_limit
        self._g_open_target = self._g_open_soft_limit
        self._g_target_force = self._g_default_force
        self._g_feedback_interval = 0.02
        self._g_feedback_elapsed = self._g_feedback_interval

    # -- lifecycle -----------------------------------------------------------

    def connect(self, enable: bool = True) -> None:
        self._arm.connect()
        try:
            if enable:
                if self._arm_control_mode == "mit":
                    self._arm_group.mode_mit()
                else:
                    self._arm_group.mode_pos_vel()
                self._arm_group.enable()

                self._gripper_group.mode_mit()
                self._gripper_group.enable()

            q, _, _ = self._wait_arm_state(timeout=5.0)
            self._endpose._q_target[:] = self._arm_q(q)
            self._read_gripper_state()

            if enable:
                self._endpose._running = True
                self._arm.start_control_loop(self._loop_cb, rate=self._arm.rate)
                self._control_active = True
                print("[RebotArm] 连接成功，电机已使能")
            else:
                print("[RebotArm] 连接成功，电机保持失能（只读模式）")

            self._connected = True
        except Exception:
            self._control_active = False
            self._connected = False
            try:
                self._arm.disconnect()
            except Exception:
                pass
            raise

    def disconnect(self, safe_home: bool = True) -> None:
        self._stop_motion()
        if safe_home and self._control_active:
            try:
                self.safe_home()
            except Exception as e:
                print(f"[RebotArm] 回零位失败: {e}")
        self._endpose._running = False
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._control_active = False
        self._connected = False
        print("[RebotArm] 已断开连接")

    def _stop_motion(self, timeout: float = 2.0) -> None:
        self._endpose._stop_send.set()
        thread = getattr(self._endpose, "_send_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # -- gripper init/status -------------------------------------------------

    def init_gripper(self) -> None:
        """夹爪由 SDK 硬件配置定义，这里仅做一次状态读取确认。"""
        self._read_gripper_state()
        print("[RebotArm] 夹爪 gripper 组已就绪，力控状态机已启用")

    @property
    def has_gripper(self) -> bool:
        return True

    @property
    def gripper_is_holding(self) -> bool:
        with self._g_lock:
            return self._g_state == _GS.HOLDING

    # -- control loop --------------------------------------------------------

    def _loop_cb(self, r, dt: float) -> None:
        try:
            with self._io_lock:
                self._endpose._loop_cb(r, dt)
                self._g_tick(dt)
        except Exception as e:
            now = time.monotonic()
            if now - self._loop_err_t > 1.0:
                print(f"[RebotArm] 控制循环异常（已忽略，1s 内不重复打印）: {e!r}")
            self._loop_err_t = now

    def _arm_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        out = np.zeros(self._n, dtype=np.float64)
        out[: min(self._n, q.size)] = q[: min(self._n, q.size)]
        return out

    def _q_for_model(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size >= self._model.nq:
            return q[: self._model.nq].copy()
        return self._pad_q_for_model(self._model, q[: self._n], self._n)

    def _read_arm_state(
        self,
        diagnostics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | tuple[np.ndarray, np.ndarray, np.ndarray, bool, list[str]]:
        q = np.zeros(self._n, dtype=np.float64)
        qd = np.zeros(self._n, dtype=np.float64)
        tau = np.zeros(self._n, dtype=np.float64)
        valid = True
        missing: list[str] = []

        with self._io_lock:
            jcfgs = list(getattr(self._arm_group, "_jcfgs", []))[: self._n]
            motors = getattr(self._arm_group, "_mm", {})
            ctrls = getattr(self._arm_group, "_cm", {})
            if len(jcfgs) != self._n:
                valid = False
                missing.append(f"groups.arm 关节数异常({len(jcfgs)}/{self._n})")

            seen: set[str] = set()
            for jc in jcfgs:
                mot = motors.get(jc.name)
                if mot is None:
                    valid = False
                    missing.append(f"{jc.name}: 未注册电机")
                    continue
                try:
                    mot.request_feedback()
                except Exception as e:
                    valid = False
                    missing.append(f"{jc.name}: 请求反馈失败({e})")
                seen.add(jc.vendor)

            for vendor in seen:
                ctrl = ctrls.get(vendor)
                if ctrl is None:
                    valid = False
                    missing.append(f"{vendor}: 控制器未初始化")
                    continue
                try:
                    ctrl.poll_feedback_once()
                except Exception as e:
                    valid = False
                    missing.append(f"{vendor}: 读取反馈失败({e})")

            for i, jc in enumerate(jcfgs):
                mot = motors.get(jc.name)
                st = mot.get_state() if mot is not None else None
                if st is None:
                    valid = False
                    missing.append(f"{jc.name}: 无反馈")
                    continue
                q[i] = float(st.pos)
                qd[i] = float(st.vel)
                tau[i] = float(st.torq)

        if diagnostics:
            return q, qd, tau, valid, missing
        return q, qd, tau, valid

    def _wait_arm_state(self, timeout: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_end = time.monotonic() + timeout
        q = qd = tau = None
        missing: list[str] = []
        while time.monotonic() < t_end:
            q, qd, tau, valid, missing = self._read_arm_state(diagnostics=True)
            if valid:
                return q, qd, tau
            time.sleep(0.02)
        detail = "；".join(dict.fromkeys(missing))
        suffix = f"：{detail}" if detail else ""
        raise RuntimeError(f"机械臂反馈未就绪{suffix}")

    # -- gripper state machine internals ------------------------------------

    def _read_gripper_state(self) -> None:
        with self._io_lock:
            try:
                self._gripper_group._request_feedback()
                jcfgs = getattr(self._gripper_group, "_jcfgs", [])
                if not jcfgs:
                    return
                mot = self._gripper_group._mm[jcfgs[0].name]
                st = mot.get_state()
                if st is not None:
                    self._g_pos = float(st.pos)
                    self._g_vel = float(st.vel)
                    self._g_torq = float(st.torq)
            except Exception:
                pass

    def _g_safe_mit(self, pos: float, vel: float, kp: float, kd: float, tau_ff: float = 0.0) -> None:
        lo = min(self._g_open_soft_limit, 0.0)
        hi = max(self._g_open_soft_limit, 0.0)
        pos_cmd = float(np.clip(pos, lo, hi))
        pos_term = kp * (pos_cmd - self._g_pos) + kd * (-self._g_vel)
        tau_safe = float(np.clip(pos_term + tau_ff, -self._g_tau_max, self._g_tau_max)) - pos_term
        with self._io_lock:
            try:
                self._gripper_group.send_mit(
                    np.array([pos_cmd], dtype=np.float64),
                    vel=np.array([vel], dtype=np.float64),
                    kp=np.array([kp], dtype=np.float64),
                    kd=np.array([kd], dtype=np.float64),
                    tau=np.array([tau_safe], dtype=np.float64),
                )
            except Exception:
                pass

    def _g_idle(self) -> None:
        with self._io_lock:
            with self._g_lock:
                self._g_state = _GS.IDLE
            self._g_safe_mit(self._g_pos, 0.0, self._g_kp_move, self._g_kd_move)

    def _require_gripper_control(self) -> None:
        if not self._control_active:
            raise RuntimeError("夹爪控制需要先调用 connect(enable=True)")

    @staticmethod
    def _step_toward(current: float, target: float, step: float) -> float:
        if current < target:
            return min(current + step, target)
        return max(current - step, target)

    def _g_tick(self, dt: float) -> None:
        with self._g_lock:
            state = self._g_state
            target_force = self._g_target_force

        if state == _GS.IDLE:
            return

        self._g_feedback_elapsed += dt
        if self._g_feedback_elapsed >= self._g_feedback_interval:
            self._g_feedback_elapsed = 0.0
            self._read_gripper_state()
        pos = self._g_pos
        vel = self._g_vel

        if state == _GS.OPENING:
            with self._g_lock:
                target = self._g_open_target
                self._g_open_q_des = self._step_toward(
                    self._g_open_q_des,
                    target,
                    abs(self._g_open_rate) * dt,
                )
                q = self._g_open_q_des
            self._g_safe_mit(q, 0.0, self._g_kp_move, self._g_kd_move)
            if abs(pos - target) < self._g_arrive_tol:
                self._g_idle()

        elif state == _GS.CLOSING:
            self._g_safe_mit(0.0, 0.0, 0.0, self._g_kd_close, self._g_close_torque)
            with self._g_lock:
                start_pos = self._g_pos_start
            if abs(pos - start_pos) >= self._g_startup_dist:
                if self._g_open_sign * pos < self._g_open_sign * self._g_hard_stop_angle:
                    self._g_idle()
                elif abs(vel) < self._g_stall_vel:
                    with self._g_lock:
                        self._g_q_contact = pos
                        self._g_contact_elapsed = 0.0
                        self._g_state = _GS.CONTACT

        elif state == _GS.CONTACT:
            with self._g_lock:
                contact_pos = self._g_q_contact
            self._g_safe_mit(contact_pos, 0.0, self._g_kp_hold, self._g_kd_hold)
            with self._g_lock:
                self._g_contact_elapsed += dt
                if self._g_contact_elapsed >= 0.02:
                    self._g_state = _GS.HOLDING

        elif state == _GS.HOLDING:
            with self._g_lock:
                contact_pos = self._g_q_contact
            self._g_safe_mit(contact_pos, 0.0, self._g_kp_hold, self._g_kd_hold, target_force)

        elif state == _GS.HOMING:
            self._g_safe_mit(0.0, 0.0, self._g_kp_move, self._g_kd_move)
            if abs(pos) < self._g_arrive_tol:
                self._g_idle()

    def _g_wait_idle(self, timeout: float = 3.0) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._g_lock:
                if self._g_state == _GS.IDLE:
                    return True
            time.sleep(0.01)
        return False

    # -- gripper public API --------------------------------------------------

    def open_gripper(self, distance_m: float = 0.09) -> None:
        self._require_gripper_control()
        self._read_gripper_state()
        d = float(np.clip(distance_m, 0.0, self._g_max_dist_m))
        raw_target = (d / self._g_max_dist_m) * self._g_angle_open
        lo = min(self._g_open_soft_limit, 0.0)
        hi = max(self._g_open_soft_limit, 0.0)
        target = float(np.clip(raw_target, lo, hi))
        with self._g_lock:
            self._g_open_target = target
            self._g_open_q_des = self._g_pos
            self._g_feedback_elapsed = self._g_feedback_interval
            self._g_state = _GS.OPENING
        self._g_wait_idle(3.0)

    def close_gripper(self) -> None:
        self._require_gripper_control()
        self._read_gripper_state()
        with self._g_lock:
            self._g_pos_start = self._g_pos
            self._g_feedback_elapsed = self._g_feedback_interval
            self._g_state = _GS.CLOSING

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        self._require_gripper_control()
        self._read_gripper_state()
        if force is not None:
            sign = 1.0 if self._g_default_force >= 0.0 else -1.0
            with self._g_lock:
                self._g_target_force = sign * float(np.clip(abs(force), 0.05, self._g_tau_max))
        with self._g_lock:
            self._g_pos_start = self._g_pos
            self._g_feedback_elapsed = self._g_feedback_interval
            self._g_state = _GS.CLOSING
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._g_lock:
                state = self._g_state
            if state == _GS.HOLDING:
                return True
            if state == _GS.IDLE:
                return False
            time.sleep(0.01)
        self._g_idle()
        return False

    def release_gripper(self, timeout: float = 4.0) -> None:
        self._require_gripper_control()
        self._read_gripper_state()
        with self._g_lock:
            self._g_open_q_des = self._g_pos
            self._g_feedback_elapsed = self._g_feedback_interval
            self._g_state = _GS.OPENING
        self._g_wait_idle(2.0)
        with self._g_lock:
            self._g_feedback_elapsed = self._g_feedback_interval
            self._g_state = _GS.HOMING
        self._g_wait_idle(timeout)

    def get_gripper_state(self) -> tuple:
        self._read_gripper_state()
        return (self._g_pos, self._g_vel, self._g_torq)

    def set_gripper_zero(self) -> bool:
        was_active = self._arm.control_loop_active
        if was_active:
            self._arm.stop_control_loop()
        ok = False
        try:
            jcfgs = getattr(self._gripper_group, "_jcfgs", [])
            if jcfgs:
                mot = self._gripper_group._mm[jcfgs[0].name]
                mot.set_zero_position()
                ok = True
                print("[RebotArm] 夹爪零点已设置")
        except Exception as e:
            print(f"[RebotArm] 夹爪零点设置失败: {e}")
        finally:
            if ok:
                self._read_gripper_state()
                self._g_idle()
            if was_active:
                self._arm.start_control_loop(self._loop_cb, rate=self._arm.rate)
        return ok

    # -- state ---------------------------------------------------------------

    def get_tcp_pose(self) -> np.ndarray:
        q, _, _, valid = self._read_arm_state()
        if not valid:
            raise RuntimeError("机械臂反馈未就绪")
        q_model = self._q_for_model(q)
        position, rotation, _ = self._compute_fk(self._model, q_model)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    # -- motion --------------------------------------------------------------

    def check_ik(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
    ) -> tuple[bool, float]:
        q, _, _, valid = self._read_arm_state()
        if not valid:
            raise RuntimeError("机械臂反馈未就绪")
        q_seed = self._arm_q(q)
        target = self._pos_rot_to_se3(
            np.array([x, y, z], dtype=np.float64),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        result = self._solve_ik(
            self._model,
            self._data,
            self._ee_frame_id,
            target,
            q_seed,
            self._ik_check_params,
            controlled_joints=self._n,
        )
        return bool(result.success), float(result.error)

    def move_to(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        if not self._control_active:
            raise RuntimeError("未连接机械臂，请先调用 connect(enable=True)")
        return bool(self._endpose.move_to_traj(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration,
        ))

    def wait_motion(self, duration: float, extra: float = 0.6) -> None:
        thread = getattr(self._endpose, "_send_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=duration + extra + 2.0)
        else:
            time.sleep(duration + extra)

    def safe_home(self, duration: float = 3.0) -> None:
        if not self._control_active:
            raise RuntimeError("未连接机械臂，请先调用 connect(enable=True)")
        self._stop_motion()

        q_state, _, _, valid = self._read_arm_state()
        if not valid:
            raise RuntimeError("机械臂反馈未就绪")
        max_err = float(np.max(np.abs(self._arm_q(q_state))))
        if max_err < 0.01:
            return
        max_vel = 2.0 * max_err / max(float(duration), 0.5)
        self._endpose.safe_home(max_vel=max_vel)

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "RebotArm":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()
