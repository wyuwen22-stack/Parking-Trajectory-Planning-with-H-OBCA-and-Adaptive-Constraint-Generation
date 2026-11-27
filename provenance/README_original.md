# 正式128例泊车规划交付

正式方案只有一套暖启动：改进精确矩形HA，原车4.7×2 m，净距5cm，每例搜索一次。节点／Dense／adaptive三个优化变体共享这条HA路径。reference_results/full128/fresh是正式结果；cached仅为历史后端隔离诊断，不是第二套正式暖启动，不用于车身模型效应比较。原案例不在128分母。

## 环境及安装
已测试Linux x86_64、Python3.13.5。使用标准CPython3.13并安装requirements.txt。未在作者电脑运行，不声称Windows环境已验收。无需原/workspace/traj_py、MATLAB源码或网络场景下载。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_fresh.py prepare
python run_fresh.py run --workers 1
python run_fresh.py summary
```

新运行写入experiments/fresh，不覆盖reference_results。可用PARKING_SAFE_OUTPUT指定新的绝对输出目录。每例有子进程硬超时、逐轮结果、断点恢复；不要同时启动两份相同来源/输出目录。原始报告计时反映实验资源，不是行业实时标准。

## 测试
```bash
python -m pytest -q -p no:cacheprovider tests/test_safe_parking.py tests/test_geometry_clearance.py tests/test_obca.py tests/test_warm_start.py tests/test_scenario.py tests/test_suite_v2.py
```
测试读取本包中的MAT导出参考，不运行或修改原MATLAB。read-only哈希文件内原绝对路径是来源标签，不是运行时依赖。tests里其余历史移植测试保留为参考，未将整目录所有测试冒称验收通过。正式已验收的44项如上。

## 结果与复现范围
reference_results/full128/fresh/{counts.json,results.csv,groups.csv,failure_layers.json}为正式全量表。各case保留共享input.npz、HA状态、三变体result.json、每轮轨迹和日志。历史cached与18例回归分开。原案例回归单列，不进入任何128分母。provenance/frozen_python_scripts保留实验执行源码；主代码只做两个移植适配：默认fresh、新运行输出独立目录。核心运动/几何/目标/求解器逐字节不变，详见portable_adapters.json。

5cm是项目规划缓冲，不是行业统一及格线。PASS_CONDITIONAL_NUMERICAL是条件数值认证，不是形式化机器证明或实车安全保证。定位、地图、跟踪误差未计。1e-12是历史比较容差，不是真实误差界。不能把线性有限采样、连续模型条件认证与实车跟踪合成一个成功率。

原论文MATLAB与原项目只读。没有新增远障奖励，once目标权重不变，没有HA fallback算OBCA成功。未加入MPC避碰或ROS2/Gazebo。未证明研究首创。文献、车辆差异与历史快照不足见PDF和PROVENANCE_SCOPE.md。

## 本次冻结验收补充
正式fresh自适应：节点100/128、历史线性采样主100/128、双5cm 50/128、新弧条件接受99/128。旧18例在正式fresh上16/18，不用历史暖启动诊断18/18补分。

已在全新隔离venv按requirements安装，并重跑上述44项；测试及安装日志在provenance。requirements.lock.txt记录此验收环境全部依赖。

逐例result.json中的warmstart旧文本为硬编码描述性标签缺陷，不能反推来源。实际正式fresh来源由ha.json、共享input.npz及全部359逐轮NPZ核对证明。原始冻结结果保持不变，详见METADATA_ERRATUM.md。
