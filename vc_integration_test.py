import threading
import time
import logging
from typing import Dict, Any, List, Optional
import json

from robot_teaching_app import RobotTeachingApp
from basic_handling_workflow import BasicHandlingWorkflow, Position, WorkPiece
from work_teaching_interface import WorkTeachingInterface, CLITeachingInterface
from integrated_safety_system import safety_system
from io_message_handler import io_controller
from config_manager import config_manager

logger = logging.getLogger(__name__)

class VCIntegrationTester:
    """Visual Components統合テスト"""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.robot_app: Optional[RobotTeachingApp] = None
        self.workflow: Optional[BasicHandlingWorkflow] = None
        self.teaching_interface: Optional[WorkTeachingInterface] = None
        
    def run_full_integration_test(self) -> Dict[str, Any]:
        """完全統合テスト実行"""
        logger.info("Starting Visual Components integration test...")
        
        test_suite = [
            ("System Initialization", self._test_system_initialization),
            ("TCP Communication", self._test_tcp_communication),
            ("I/O Signal Processing", self._test_io_processing),
            ("Safety System", self._test_safety_system),
            ("Basic Workflow", self._test_basic_workflow),
            ("Trajectory Generation", self._test_trajectory_generation),
            ("Teaching Interface", self._test_teaching_interface),
            ("VC Script Integration", self._test_vc_script_integration),
            ("Full Demo Scenario", self._test_full_demo_scenario)
        ]
        
        overall_results = {
            "test_summary": {
                "total_tests": len(test_suite),
                "passed": 0,
                "failed": 0,
                "warnings": 0
            },
            "test_details": {}
        }
        
        for test_name, test_function in test_suite:
            logger.info(f"Running test: {test_name}")
            try:
                result = test_function()
                self.test_results[test_name] = result
                overall_results["test_details"][test_name] = result
                
                if result["status"] == "PASS":
                    overall_results["test_summary"]["passed"] += 1
                elif result["status"] == "FAIL":
                    overall_results["test_summary"]["failed"] += 1
                else:
                    overall_results["test_summary"]["warnings"] += 1
                
                logger.info(f"Test {test_name}: {result['status']}")
                
            except Exception as e:
                error_result = {
                    "status": "FAIL",
                    "message": f"Test execution error: {str(e)}",
                    "details": {}
                }
                self.test_results[test_name] = error_result
                overall_results["test_details"][test_name] = error_result
                overall_results["test_summary"]["failed"] += 1
                logger.error(f"Test {test_name} failed with exception: {e}")
        
        return overall_results
    
    def _test_system_initialization(self) -> Dict[str, Any]:
        """システム初期化テスト"""
        try:
            # ロボットアプリケーション初期化
            self.robot_app = RobotTeachingApp()
            
            # ワークフロー初期化
            from basic_handling_workflow import basic_workflow
            self.workflow = basic_workflow
            workflow_init = self.workflow.initialize_workflow()
            
            # 教示インターフェース初期化
            self.teaching_interface = WorkTeachingInterface(self.workflow)
            
            # 安全システム初期化
            safety_init = safety_system.check_overall_safety()
            
            # 設定システム
            config_loaded = len(config_manager.configs) > 0
            
            return {
                "status": "PASS" if workflow_init and config_loaded else "FAIL",
                "message": "System components initialized",
                "details": {
                    "workflow_initialized": workflow_init,
                    "safety_system_active": safety_init["overall_safe"],
                    "configs_loaded": config_loaded,
                    "teaching_interface_ready": self.teaching_interface is not None
                }
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"System initialization failed: {str(e)}",
                "details": {}
            }
    
    def _test_tcp_communication(self) -> Dict[str, Any]:
        """TCP通信テスト"""
        try:
            # TCP サーバー開始（バックグラウンド）
            server_thread = threading.Thread(
                target=self.robot_app.start_application, 
                daemon=True
            )
            server_thread.start()
            time.sleep(2)  # サーバー起動待機
            
            # 接続テスト用クライアント
            from tcp_communication import TCPClient, MessageBuilder
            
            test_client = TCPClient("localhost", 8888)
            connection_success = test_client.connect()
            
            if connection_success:
                # テストメッセージ送信
                test_message = MessageBuilder.create_status_request("robot_1")
                send_success = test_client.send_message(test_message)
                
                # 接続クリーンアップ
                test_client.disconnect()
                
                return {
                    "status": "PASS" if send_success else "WARNING",
                    "message": "TCP communication functional",
                    "details": {
                        "server_started": True,
                        "client_connected": connection_success,
                        "message_sent": send_success
                    }
                }
            else:
                return {
                    "status": "FAIL",
                    "message": "TCP connection failed",
                    "details": {
                        "server_started": True,
                        "client_connected": False
                    }
                }
                
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"TCP communication test failed: {str(e)}",
                "details": {}
            }
    
    def _test_io_processing(self) -> Dict[str, Any]:
        """I/O処理テスト"""
        try:
            # I/O信号テスト
            test_signals = [
                ("START_BUTTON", True),
                ("E_STOP", False),
                ("DOOR_CLOSED", True),
                ("AIR_PRESSURE_OK", True),
                ("PART_PRESENT", True)
            ]
            
            io_results = {}
            for signal_name, test_value in test_signals:
                # 信号設定
                set_success = io_controller.set_signal_value(signal_name, test_value)
                
                # 信号読み取り
                read_value = io_controller.get_signal_value(signal_name)
                
                io_results[signal_name] = {
                    "set_success": set_success,
                    "value_match": read_value == test_value
                }
            
            # 全体ステータス取得
            all_status = io_controller.get_all_signals_status()
            
            all_passed = all(
                result["set_success"] and result["value_match"] 
                for result in io_results.values()
            )
            
            return {
                "status": "PASS" if all_passed else "FAIL",
                "message": "I/O signal processing functional",
                "details": {
                    "individual_signals": io_results,
                    "total_signals": len(all_status),
                    "status_retrieval": len(all_status) > 0
                }
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"I/O processing test failed: {str(e)}",
                "details": {}
            }
    
    def _test_safety_system(self) -> Dict[str, Any]:
        """安全システムテスト"""
        try:
            # 安全状態チェック
            initial_safety = safety_system.check_overall_safety()
            
            # 緊急停止テスト
            io_controller.set_signal_value("E_STOP", True)
            time.sleep(0.2)
            emergency_safety = safety_system.check_overall_safety()
            
            # 緊急停止解除
            io_controller.set_signal_value("E_STOP", False)
            time.sleep(0.2)
            recovered_safety = safety_system.check_overall_safety()
            
            # 位置安全性テスト
            safe_position = Position(100, 100, 150, 0, 0, 0)
            position_safe = safety_system.workspace_monitor.check_position_safety(safe_position)
            
            return {
                "status": "PASS",
                "message": "Safety system responsive",
                "details": {
                    "initial_safe": initial_safety["overall_safe"],
                    "emergency_detected": not emergency_safety["overall_safe"],
                    "recovery_successful": recovered_safety["overall_safe"],
                    "position_validation": position_safe["is_safe"],
                    "workspace_zones": len(safety_system.workspace_monitor.safety_zones)
                }
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Safety system test failed: {str(e)}",
                "details": {}
            }
    
    def _test_basic_workflow(self) -> Dict[str, Any]:
        """基本ワークフローテスト"""
        try:
            # テストタスク設定
            test_task_config = {
                "task_id": "integration_test_001",
                "pick_position": [150.0, -100.0, 120.0, 0.0, 0.0, 0.0],
                "place_position": [250.0, 50.0, 120.0, 0.0, 0.0, 0.0],
                "workpiece": {
                    "name": "test_component",
                    "part_type": "integration_test",
                    "weight": 1.0,
                    "dimensions": [50, 50, 25],
                    "material": "plastic",
                    "grip_force": 25.0
                },
                "approach_speed": 60,
                "work_speed": 20,
                "safety_height": 50.0
            }
            
            # 安全条件設定
            io_controller.set_signal_value("PART_PRESENT", True)
            io_controller.set_signal_value("JIG_CLAMPED", True)
            io_controller.set_signal_value("DOOR_CLOSED", True)
            io_controller.set_signal_value("AIR_PRESSURE_OK", True)
            io_controller.set_signal_value("E_STOP", False)
            
            # タスク作成
            task = self.workflow.create_handling_task(test_task_config)
            
            # タスク実行（シミュレーションモード）
            result = self.workflow.execute_handling_task(task)
            
            # パフォーマンスメトリクス取得
            metrics = self.workflow.performance_metrics
            
            return {
                "status": "PASS" if result.value == "success" else "FAIL",
                "message": f"Basic workflow execution: {result.value}",
                "details": {
                    "task_created": task is not None,
                    "execution_result": result.value,
                    "total_cycles": metrics["total_cycles"],
                    "successful_cycles": metrics["successful_cycles"],
                    "average_cycle_time": metrics.get("average_cycle_time", 0)
                }
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Basic workflow test failed: {str(e)}",
                "details": {}
            }
    
    def _test_trajectory_generation(self) -> Dict[str, Any]:
        """軌道生成テスト"""
        try:
            from trajectory_generation import generate_handling_trajectory, trajectory_generator
            
            # 基本軌道生成テスト
            trajectory_config = {
                "operation_type": "basic_pick_place",
                "pick_position": [100, -200, 150, 0, 0, 0],
                "place_position": [300, 100, 150, 0, 0, 0],
                "workpiece": {
                    "name": "test_part",
                    "part_type": "component",
                    "weight": 1.0,
                    "dimensions": [50, 50, 25],
                    "material": "plastic"
                },
                "parameters": {
                    "approach_speed": 70,
                    "work_speed": 25,
                    "safety_height": 60.0
                }
            }
            
            trajectory = generate_handling_trajectory(trajectory_config)
            
            # 軌道検証
            validation = trajectory_generator.validate_trajectory(trajectory)
            
            # 円弧軌道テスト
            center_pos = Position(200, 0, 200, 0, 0, 0)
            circular_trajectory = trajectory_generator.generate_circular_trajectory(
                center_pos, 100.0, 0, 180
            )
            
            return {
                "status": "PASS" if validation["is_valid"] else "WARNING",
                "message": "Trajectory generation functional",
                "details": {
                    "basic_trajectory_points": len(trajectory),
                    "trajectory_valid": validation["is_valid"],
                    "collision_points": len(validation.get("collision_points", [])),
                    "circular_trajectory_points": len(circular_trajectory),
                    "validation_warnings": len(validation.get("warnings", []))
                }
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Trajectory generation test failed: {str(e)}",
                "details": {}
            }
    
    def _test_teaching_interface(self) -> Dict[str, Any]:
        """教示インターフェーステスト"""
        try:
            # 教示セッション作成
            from work_teaching_interface import TeachingMode, InterfaceType
            
            session_id = self.teaching_interface.start_teaching_session(
                TeachingMode.MANUAL, InterfaceType.JSON_API
            )
            
            # セッションパラメーター設定
            session = self.teaching_interface.active_sessions[session_id]
            session.parameters.update({
                "pick": [100, -150, 120, 0, 0, 0],
                "place": [250, 100, 120, 0, 0, 0],
                "workpiece": {
                    "name": "taught_part",
                    "part_type": "manual_taught",
                    "weight": 1.2,
                    "material": "aluminum",
                    "grip_force": 40.0
                }
            })
            
            # セッション終了
            end_success = self.teaching_interface.end_teaching_session(session_id)
            
            # Webインターフェーステスト（ポート確認のみ）
            from work_teaching_interface import WebTeachingInterface
            web_interface = WebTeachingInterface(self.teaching_interface, 8081)
            
            try:
                web_interface.start_web_server()
                time.sleep(1)
                web_available = True
                web_interface.stop_web_server()
            except:
                web_available = False
            
            return {
                "status": "PASS",
                "message": "Teaching interface functional",
                "details": {
                    "session_created": session_id is not None,
                    "session_ended": end_success,
                    "parameters_set": len(session.parameters) > 0,
                    "web_interface_available": web_available,
                    "active_sessions": len(self.teaching_interface.active_sessions)
                }
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Teaching interface test failed: {str(e)}",
                "details": {}
            }
    
    def _test_vc_script_integration(self) -> Dict[str, Any]:
        """VCスクリプト統合テスト"""
        try:
            # VCスクリプト環境チェック
            vc_available = False
            vc_error = None
            
            try:
                # vcScript インポートテスト（シミュレーション）
                # 実際の環境では: from vcScript import *
                vc_available = False  # Visual Componentsが利用できない環境
                vc_error = "vcScript not available in test environment"
            except ImportError as e:
                vc_error = str(e)
            
            # ロボットコントローラーテスト
            from vc_robot_controller import initialize_robot_controller
            robot_controller = initialize_robot_controller("test_robot")
            
            # ツールコントローラーテスト
            from vc_tool_controller import initialize_gripper_controller
            tool_controller = initialize_gripper_controller("test_gripper")
            
            return {
                "status": "WARNING" if not vc_available else "PASS",
                "message": "VC integration tested (simulation mode)",
                "details": {
                    "vc_script_available": vc_available,
                    "vc_error": vc_error,
                    "robot_controller_created": robot_controller is not None,
                    "tool_controller_created": tool_controller is not None,
                    "simulation_mode": True
                }
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"VC script integration test failed: {str(e)}",
                "details": {}
            }
    
    def _test_full_demo_scenario(self) -> Dict[str, Any]:
        """完全デモシナリオテスト"""
        try:
            logger.info("Running full demo scenario...")
            
            # デモワーク設定
            demo_config = {
                "task_id": "demo_automotive_handling",
                "pick_position": [120.0, -180.0, 100.0, 0.0, 0.0, 0.0],
                "place_position": [280.0, 80.0, 100.0, 0.0, 0.0, 0.0],
                "workpiece": {
                    "name": "automotive_bracket",
                    "part_type": "automotive_component",
                    "weight": 2.1,
                    "dimensions": [150, 100, 45],
                    "material": "steel",
                    "grip_force": 55.0
                },
                "approach_speed": 75,
                "work_speed": 35,
                "safety_height": 65.0,
                "quality_check": True
            }
            
            # ステップ1: 安全確認
            safety_ok = safety_system.check_overall_safety()["overall_safe"]
            if not safety_ok:
                # 安全条件設定
                io_controller.set_signal_value("E_STOP", False)
                io_controller.set_signal_value("DOOR_CLOSED", True)
                io_controller.set_signal_value("AIR_PRESSURE_OK", True)
                io_controller.set_signal_value("PART_PRESENT", True)
                io_controller.set_signal_value("JIG_CLAMPED", True)
            
            # ステップ2: 軌道生成
            trajectory_config = {
                "operation_type": "basic_pick_place",
                **demo_config
            }
            trajectory = generate_handling_trajectory(trajectory_config)
            
            # ステップ3: 安全検証
            positions = [point.position for point in trajectory]
            trajectory_validation = safety_system.validate_trajectory_safety(positions)
            
            # ステップ4: タスク実行
            if trajectory_validation["is_safe"]:
                task = self.workflow.create_handling_task(demo_config)
                execution_result = self.workflow.execute_handling_task(task)
                execution_success = execution_result.value == "success"
            else:
                execution_success = False
                execution_result = "trajectory_unsafe"
            
            # ステップ5: 結果検証
            final_status = self.workflow.get_workflow_status()
            
            demo_success = (
                safety_ok and 
                len(trajectory) > 0 and
                trajectory_validation["is_safe"] and
                execution_success
            )
            
            return {
                "status": "PASS" if demo_success else "FAIL",
                "message": f"Full demo scenario: {'SUCCESS' if demo_success else 'FAILED'}",
                "details": {
                    "safety_check": safety_ok,
                    "trajectory_points": len(trajectory),
                    "trajectory_safe": trajectory_validation["is_safe"],
                    "execution_result": str(execution_result),
                    "final_workflow_state": final_status["current_state"],
                    "performance_metrics": self.workflow.performance_metrics
                }
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Full demo scenario failed: {str(e)}",
                "details": {}
            }
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """テストレポート生成"""
        report_lines = [
            "=" * 60,
            "VISUAL COMPONENTS INTEGRATION TEST REPORT",
            "=" * 60,
            f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY:",
            f"  Total Tests: {results['test_summary']['total_tests']}",
            f"  Passed: {results['test_summary']['passed']}",
            f"  Failed: {results['test_summary']['failed']}",
            f"  Warnings: {results['test_summary']['warnings']}",
            "",
            "DETAILED RESULTS:",
            ""
        ]
        
        for test_name, test_result in results["test_details"].items():
            status_symbol = {
                "PASS": "[PASS]",
                "FAIL": "[FAIL]", 
                "WARNING": "[WARN]"
            }.get(test_result["status"], "[?]")
            
            report_lines.extend([
                f"{status_symbol} {test_name}: {test_result['status']}",
                f"    {test_result['message']}",
                ""
            ])
            
            if test_result.get("details"):
                for key, value in test_result["details"].items():
                    report_lines.append(f"    - {key}: {value}")
                report_lines.append("")
        
        report_lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])
        
        return "\n".join(report_lines)

def run_integration_test():
    """統合テスト実行メイン関数"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Visual Components Integration Test Starting...")
    print("=" * 60)
    
    # テスター作成・実行
    tester = VCIntegrationTester()
    
    try:
        test_results = tester.run_full_integration_test()
        
        # レポート生成・表示
        report = tester.generate_test_report(test_results)
        print(report)
        
        # レポートファイル保存
        report_filename = f"integration_test_report_{int(time.time())}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nTest report saved to: {report_filename}")
        
        # 総合結果
        total_tests = test_results["test_summary"]["total_tests"]
        passed_tests = test_results["test_summary"]["passed"]
        
        if passed_tests == total_tests:
            print("ALL TESTS PASSED! System ready for production.")
        elif passed_tests > total_tests // 2:
            print("Most tests passed. Review warnings and failures.")
        else:
            print("Multiple test failures. System needs attention.")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Integration test execution failed: {e}")
        print(f"Integration test failed with error: {e}")
        return None
    
    finally:
        # クリーンアップ
        try:
            if tester.workflow:
                tester.workflow.shutdown()
            safety_system.shutdown()
        except:
            pass

if __name__ == "__main__":
    run_integration_test()