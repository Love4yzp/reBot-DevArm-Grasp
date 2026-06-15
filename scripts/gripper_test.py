#!/usr/bin/env python3
"""Minimal gripper motion test using reBotArm_control_py directly.

Usage:
  unset PYTHONPATH
  python scripts/gripper_test.py
  python scripts/gripper_test.py --target 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk" / "reBotArm_control_py"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from motorbridge import Controller, Mode  # noqa: E402


def _gripper_motor(rebotarm: RebotArm):
    joints = getattr(rebotarm.gripper, "_jcfgs", [])
    if not joints:
        raise RuntimeError("硬件配置没有 gripper 关节")
    return rebotarm.gripper._mm[joints[0].name], joints[0]


def _read_gripper(rebotarm: RebotArm) -> tuple[float, float, float]:
    rebotarm.gripper._request_feedback()
    motor, _ = _gripper_motor(rebotarm)
    state = motor.get_state()
    if state is None:
        raise RuntimeError("夹爪反馈未就绪")
    return float(state.pos), float(state.vel), float(state.torq)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reBotArm gripper enable/motion test")
    parser.add_argument("--target", type=float, default=None, help="夹爪 MIT 目标角度，单位 rad")
    parser.add_argument("--mode", choices=("mit", "tau", "posvel", "loop-mit", "loop-tau"), default="mit")
    parser.add_argument("--tau", type=float, default=-0.2, help="--mode tau 时发送的力矩")
    parser.add_argument("--vlim", type=float, default=0.5, help="--mode posvel 时的位置速度限制")
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--hw-yaml", default=None, help="可选：指定 SDK config 下的硬件 yaml")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--motor-id", type=lambda x: int(x, 0), default=0x07)
    parser.add_argument("--feedback-id", type=lambda x: int(x, 0), default=None, help="直接用指定 host id 测试 RobStride 夹爪")
    parser.add_argument("--loc-kp", type=float, default=10.0, help="RobStride 原生位置模式的位置环增益")
    parser.add_argument("--no-direct-enable", action="store_true", help="不直接调用 gripper motor.enable()")
    return parser.parse_args()


def _direct_read(motor) -> tuple[float, float, float]:
    motor.request_feedback()
    state = motor.get_state()
    if state is None:
        raise RuntimeError("夹爪反馈未就绪")
    return float(state.pos), float(state.vel), float(state.torq)


def _direct_test(args: argparse.Namespace) -> int:
    feedback_id = int(args.feedback_id)
    target = args.target if args.target is not None else 1.0
    print(
        f"[gripper_test/direct] channel={args.channel} motor_id={args.motor_id:#x} "
        f"feedback_id={feedback_id:#x}"
    )
    ctrl = Controller(args.channel)
    motor = ctrl.add_robstride_motor(args.motor_id, feedback_id, "rs-00")
    try:
        device_id, responder_id = motor.robstride_ping_host_id(feedback_id, 500)
        print(f"[gripper_test/direct] ping_host_id device_id={device_id:#x} responder_id={responder_id:#x}")
        try:
            run_mode = motor.robstride_get_param_i8(0x7005, 500)
            fault_raw, warning_raw = motor.robstride_get_fault_report()
            print(
                f"[gripper_test/direct] before run_mode={run_mode} "
                f"fault=0x{fault_raw:08x} warning=0x{warning_raw:08x}"
            )
        except Exception as exc:
            print(f"[gripper_test/direct] read status failed: {exc}")

        if args.mode == "posvel":
            print("[gripper_test/direct] ensure POS_VEL")
            ctrl.disable_all()
            time.sleep(0.06)
            motor.ensure_mode(Mode.POS_VEL, 1000)
        else:
            print("[gripper_test/direct] ensure MIT")
            ctrl.disable_all()
            time.sleep(0.06)
            motor.ensure_mode(Mode.MIT, 1000)
        try:
            run_mode = motor.robstride_get_param_i8(0x7005, 500)
            print(f"[gripper_test/direct] after ensure run_mode={run_mode}")
        except Exception as exc:
            print(f"[gripper_test/direct] read run_mode failed: {exc}")

        ctrl.enable_all()
        motor.enable()
        time.sleep(0.2)
        pos0, vel0, torq0 = _direct_read(motor)
        print(f"[gripper_test/direct] state0 pos={pos0:+.4f} vel={vel0:+.4f} torq={torq0:+.4f}")

        steps = max(2, int(args.duration / 0.02))
        if args.mode == "posvel":
            print(
                f"[gripper_test/direct] native pos target={target:+.4f} "
                f"vlim={args.vlim:+.4f} loc_kp={args.loc_kp:+.4f}"
            )
            motor.robstride_write_param_f32(0x7017, abs(float(args.vlim)))
            motor.robstride_write_param_f32(0x701E, float(args.loc_kp))
            for i in range(steps):
                motor.robstride_write_param_f32(0x7016, float(target))
                if i % 10 == 0:
                    pos, vel, torq = _direct_read(motor)
                    print(f"\r  direct pos={pos:+.4f} vel={vel:+.4f} torq={torq:+.4f}", end="", flush=True)
                time.sleep(0.02)
            print()
        elif args.mode == "tau":
            print(f"[gripper_test/direct] apply tau={args.tau:+.4f}")
            for i in range(steps):
                motor.send_mit(pos0, 0.0, 0.0, args.kd, args.tau)
                if i % 10 == 0:
                    pos, vel, torq = _direct_read(motor)
                    print(f"\r  direct tau pos={pos:+.4f} vel={vel:+.4f} torq={torq:+.4f}", end="", flush=True)
                time.sleep(0.02)
            print()
        else:
            print(f"[gripper_test/direct] mit target={target:+.4f}")
            for i in range(steps):
                s = (i + 1) / steps
                cmd = pos0 + (target - pos0) * s
                motor.send_mit(cmd, 0.0, args.kp, args.kd, 0.0)
                if i % 10 == 0:
                    pos, vel, torq = _direct_read(motor)
                    print(f"\r  direct mit pos={pos:+.4f} vel={vel:+.4f} torq={torq:+.4f}", end="", flush=True)
                time.sleep(0.02)
            print()

        pos1, vel1, torq1 = _direct_read(motor)
        print(f"[gripper_test/direct] state1 pos={pos1:+.4f} vel={vel1:+.4f} torq={torq1:+.4f}")
    finally:
        try:
            ctrl.disable_all()
        except Exception:
            pass
        motor.close()
        ctrl.close()
        print("[gripper_test/direct] disconnected")
    return 0


def _loop_test(args: argparse.Namespace) -> int:
    rebotarm = RebotArm(args.hw_yaml)
    target = args.target
    if target is None:
        target = 1.0 if "rs" in rebotarm.hardware_yaml.lower() else -1.0

    print(f"[gripper_test/loop] hardware_yaml={rebotarm.hardware_yaml}")
    rebotarm.connect()
    running = {"active": True}
    command = {"target": 0.0, "tau": 0.0, "pos0": 0.0}

    def loop_cb(r: RebotArm, dt: float) -> None:
        if not running["active"]:
            return
        if args.mode == "loop-tau":
            r.gripper.send_mit(
                pos=np.array([command["pos0"]], dtype=np.float64),
                vel=np.zeros(1),
                kp=np.zeros(1),
                kd=np.array([args.kd], dtype=np.float64),
                tau=np.array([command["tau"]], dtype=np.float64),
            )
        else:
            r.gripper.send_mit(
                pos=np.array([command["target"]], dtype=np.float64),
                vel=np.zeros(1),
                kp=np.array([args.kp], dtype=np.float64),
                kd=np.array([args.kd], dtype=np.float64),
                tau=np.zeros(1),
            )

    try:
        motor, joint_cfg = _gripper_motor(rebotarm)
        print(
            f"[gripper_test/loop] joint name={joint_cfg.name} vendor={joint_cfg.vendor} "
            f"motor_id={joint_cfg.motor_id:#x} feedback_id={joint_cfg.feedback_id:#x}"
        )
        rebotarm.gripper.mode_mit()
        rebotarm.gripper.enable()
        if not args.no_direct_enable:
            motor.enable()
        time.sleep(0.2)
        pos0, vel0, torq0 = _read_gripper(rebotarm)
        command["pos0"] = pos0
        command["target"] = pos0
        print(f"[gripper_test/loop] state0 pos={pos0:+.4f} vel={vel0:+.4f} torq={torq0:+.4f}")

        rebotarm.start_control_loop(loop_cb, rate=rebotarm.rate)
        print(f"[gripper_test/loop] control loop started @ {rebotarm.rate} Hz")

        steps = max(2, int(args.duration / 0.02))
        if args.mode == "loop-tau":
            command["tau"] = float(args.tau)
            print(f"[gripper_test/loop] apply tau={args.tau:+.4f}")
            for i in range(steps):
                if i % 10 == 0:
                    pos, vel, torq = _read_gripper(rebotarm)
                    print(f"\r  loop tau pos={pos:+.4f} vel={vel:+.4f} torq={torq:+.4f}", end="", flush=True)
                time.sleep(0.02)
            print()
        else:
            print(f"[gripper_test/loop] move to target={target:+.4f} rad")
            for i in range(steps):
                s = (i + 1) / steps
                command["target"] = pos0 + (target - pos0) * s
                if i % 10 == 0:
                    pos, vel, torq = _read_gripper(rebotarm)
                    print(f"\r  loop mit pos={pos:+.4f} vel={vel:+.4f} torq={torq:+.4f}", end="", flush=True)
                time.sleep(0.02)
            print()

        pos1, vel1, torq1 = _read_gripper(rebotarm)
        print(f"[gripper_test/loop] state1 pos={pos1:+.4f} vel={vel1:+.4f} torq={torq1:+.4f}")
    finally:
        running["active"] = False
        try:
            rebotarm.stop_control_loop()
        except Exception:
            pass
        rebotarm.disconnect()
        print("[gripper_test/loop] disconnected")
    return 0


def main() -> int:
    args = parse_args()
    if args.feedback_id is not None:
        return _direct_test(args)
    if args.mode in ("loop-mit", "loop-tau"):
        return _loop_test(args)

    rebotarm = RebotArm(args.hw_yaml)
    print(f"[gripper_test] hardware_yaml={rebotarm.hardware_yaml}")
    print(f"[gripper_test] has_gripper={rebotarm.has_gripper}")
    if not rebotarm.has_gripper:
        print("[gripper_test] 当前硬件配置没有 gripper 组")
        return 1

    target = args.target
    if target is None:
        target = 1.0 if "rs" in rebotarm.hardware_yaml.lower() else -1.0

    rebotarm.connect()
    try:
        print("[gripper_test] connected")
        motor, joint_cfg = _gripper_motor(rebotarm)
        print(
            f"[gripper_test] joint name={joint_cfg.name} vendor={joint_cfg.vendor} "
            f"motor_id={joint_cfg.motor_id:#x} feedback_id={joint_cfg.feedback_id:#x}"
        )
        if joint_cfg.vendor == "robstride":
            try:
                device_id, responder_id = motor.robstride_ping()
                print(f"[gripper_test] robstride_ping device_id={device_id:#x} responder_id={responder_id:#x}")
            except Exception as exc:
                print(f"[gripper_test] robstride_ping failed: {exc}")

        if args.mode == "posvel":
            print("[gripper_test] gripper.mode_pos_vel()")
            mode_ok = rebotarm.gripper.mode_pos_vel()
            print(f"[gripper_test] mode_pos_vel ok={mode_ok}")
        else:
            print("[gripper_test] gripper.mode_mit()")
            mode_ok = rebotarm.gripper.mode_mit()
            print(f"[gripper_test] mode_mit ok={mode_ok}")

        print("[gripper_test] gripper.enable()")
        rebotarm.gripper.enable()
        if not args.no_direct_enable:
            print("[gripper_test] motor.enable()")
            motor.enable()
        time.sleep(0.2)

        pos0, vel0, torq0 = _read_gripper(rebotarm)
        print(f"[gripper_test] state0 pos={pos0:+.4f} vel={vel0:+.4f} torq={torq0:+.4f}")

        steps = max(2, int(args.duration / 0.02))
        if args.mode == "tau":
            print(f"[gripper_test] apply tau={args.tau:+.4f}")
            for i in range(steps):
                rebotarm.gripper.send_mit(
                    pos=np.array([pos0], dtype=np.float64),
                    vel=np.zeros(1),
                    kp=np.zeros(1),
                    kd=np.array([args.kd], dtype=np.float64),
                    tau=np.array([args.tau], dtype=np.float64),
                )
                if i % 10 == 0:
                    pos, vel, torq = _read_gripper(rebotarm)
                    print(f"\r  tau pos={pos:+.4f} vel={vel:+.4f} torq={torq:+.4f}", end="", flush=True)
                time.sleep(0.02)
            print()
        elif args.mode == "posvel":
            print(f"[gripper_test] send_pos_vel target={target:+.4f} rad vlim={args.vlim:+.4f}")
            for i in range(steps):
                rebotarm.gripper.send_pos_vel(
                    pos=np.array([target], dtype=np.float64),
                    vlim=np.array([args.vlim], dtype=np.float64),
                )
                if i % 10 == 0:
                    pos, vel, torq = _read_gripper(rebotarm)
                    print(f"\r  posvel pos={pos:+.4f} vel={vel:+.4f} torq={torq:+.4f}", end="", flush=True)
                time.sleep(0.02)
            print()
        else:
            print(f"[gripper_test] move to target={target:+.4f} rad")
            for i in range(steps):
                s = (i + 1) / steps
                cmd = pos0 + (target - pos0) * s
                rebotarm.gripper.send_mit(
                    pos=np.array([cmd], dtype=np.float64),
                    vel=np.zeros(1),
                    kp=np.array([args.kp], dtype=np.float64),
                    kd=np.array([args.kd], dtype=np.float64),
                    tau=np.zeros(1),
                )
                if i % 10 == 0:
                    pos, vel, torq = _read_gripper(rebotarm)
                    print(f"\r  moving pos={pos:+.4f} vel={vel:+.4f} torq={torq:+.4f}", end="", flush=True)
                time.sleep(0.02)
            print()

            t_end = time.monotonic() + args.hold
            while time.monotonic() < t_end:
                rebotarm.gripper.send_mit(
                    pos=np.array([target], dtype=np.float64),
                    vel=np.zeros(1),
                    kp=np.array([args.kp], dtype=np.float64),
                    kd=np.array([args.kd], dtype=np.float64),
                    tau=np.zeros(1),
                )
                time.sleep(0.02)
        pos1, vel1, torq1 = _read_gripper(rebotarm)
        print(f"[gripper_test] state1 pos={pos1:+.4f} vel={vel1:+.4f} torq={torq1:+.4f}")

        if args.mode != "tau":
            print("[gripper_test] return to 0.0 rad")
            if args.mode == "posvel":
                rebotarm.gripper.send_pos_vel(
                    pos=np.array([0.0], dtype=np.float64),
                    vlim=np.array([args.vlim], dtype=np.float64),
                )
                time.sleep(args.hold)
            else:
                for i in range(steps):
                    s = (i + 1) / steps
                    cmd = pos1 + (0.0 - pos1) * s
                    rebotarm.gripper.send_mit(
                        pos=np.array([cmd], dtype=np.float64),
                        vel=np.zeros(1),
                        kp=np.array([args.kp], dtype=np.float64),
                        kd=np.array([args.kd], dtype=np.float64),
                        tau=np.zeros(1),
                    )
                    time.sleep(0.02)

        pos2, vel2, torq2 = _read_gripper(rebotarm)
        print(f"[gripper_test] state2 pos={pos2:+.4f} vel={vel2:+.4f} torq={torq2:+.4f}")
    finally:
        rebotarm.disconnect()
        print("[gripper_test] disconnected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
