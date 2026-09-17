# Fly Trader（IEX 实时信号，不下单）

这是一个 Windows 上运行的研究实验：从 Alpaca 免费行情 WebSocket 读取 IEX 实时成交与 overnight 实时指示性报价，以公开的 MaleCNS 连接组运行简化 LIF 神经动力学，并记录 BUY/HOLD/SELL **研究信号**。项目没有创建交易客户端，也没有任何下单入口。旧的老虎延时行情入口保留为显式 `--source tiger` 兼容模式。

当前冻结实验为 `fly-core-v1`。系统同时记录三条互不覆盖的轨道：`fly_raw` 是果蝇原始输出并作为主指标，`fly_filtered` 是人工平滑与阈值对照，`risk_executable` 是经过只读风控后的可执行性对照。详细规则见 [冻结实验规范](docs/experiment-protocol.md)。

默认观察美股 `NVDA`、港股 `700.HK`（腾讯）和 `2513.HK`（智谱）。三只股票共享一份只读连接矩阵，但分别维护膜电位、放电、行情刺激、随机数状态和信号滤波器，因此不会把不同股票输入混入同一个神经状态。模型始终保留上游原始 20ms 动力学步长。三只标的均进入阶段三订单提案硬白名单，但系统仍只生成提案，不提交订单。

订单提案会逐项检查常规连续交易时段、行情新鲜度、模拟账户状态、2% 日损失熔断、只做多、现金、10% 单标的仓位、70% 总仓位和 5 分钟提案冷却。仪表盘显示每项检查结果；所有逻辑仍为只读预演，不调用任何券商下单接口。

美股和港股共用 `StrategyPipeline` 处理神经读出、信号过滤、只读风控及版本化 JSONL 事件，避免两个市场各自维护不同的决策逻辑。当前统一事件版本为 schema 8；事件内保存实验编号、规范指纹和三条轨道。

港股神经模型与美股一样按持续 50Hz 时钟推进，不再随报价到达间隔批量运行。报价超过 2 秒后刺激逐步衰减，超过 5 秒强制观望。

## 数据流与映射

每个 IEX 成交时间戳只处理一次。去重键是 `(symbol, exchange timestamp)`；相同时间戳和迟到的旧事件都会被丢弃。MaleCNS 独立以 50Hz（20ms）连续运行，最新成交更新持续刺激；13 tick（约 260ms）滚动窗负责神经活动解码。行情来得比模型消费速度快时只保留最新状态，不积压陈旧刺激。

IEX 流只覆盖 IEX 单一交易所，不代表全美 SIP 综合行情。进程启动后看到的第一笔成交作为本次连接的参考价和开盘代理；高、低、收和成交量由随后收到的 IEX trades 累积。

隔夜同时订阅 `v1beta1/overnight` 的免费实时指示性 quotes。使用有效 bid/ask 的中间价作为刺激价，拒绝零价或交叉报价，不虚构成交量。日志以 `feed: "overnight"`、`event_type: "indicative_quote"` 明确标注；它不是实际成交价。IEX 则记录为 `feed: "iex"`、`event_type: "trade"`。

为抑制小型下降神经池产生的离散抖动，解码器先计算最近 1 秒的原始分数均值，再以 0.35 秒半衰期做 EMA。每只股票独立预热 5 分钟，并至少积累美股 100 个、港股 30 个有效样本；随后用最近 30 分钟绝对平滑分数的第 95 百分位作为动态进入阈值（限制在 2 到 20），退出阈值为进入阈值的 40%。预热期间强制观望，状态变化后有 1 秒冷却期；订单提案另要求稳定信号持续 3 秒。页面和 JSONL 会记录动态阈值、剩余预热时间、基线样本数以及距离触发值。隔夜报价点差超过 100bp、买卖价交叉或相邻中间价跳变超过 2% 时直接丢弃。最新行情超过 2 秒后，神经刺激在接下来的 3 秒线性衰减；超过 5 秒强制观望。

终端不会逐条打印全部行情：稳定信号变化时立即提示，其他时间每 5 秒输出一次最新价格、原始/稳定信号、神经分数和期间行情数量。每条合格行情仍完整写入 JSONL。

行情到神经刺激的映射是固定且可审计的：

| MaleCNS 视觉受体连续分区 | 行情特征 | 归一化 |
|---|---|---|
| 0 | `close / reference_price` 对数收益 | 2% 映射到 ±1，再映射到 0..1 |
| 1 | 上述收益的相反数 | 同上 |
| 2 | `close / open` 对数收益 | 同上 |
| 3 | 上述收益的相反数 | 同上 |
| 4 | 收盘价在日内高低区间的位置 | -1..1 映射到 0..1 |
| 5 | 日内振幅 + 相邻快照成交量对数变化绝对值 | 截断到 0..1 |

停牌时所有外部视觉刺激归零。神经动力学直接复用 ornata/fly 的参数：`dt=20ms`、膜时间常数 `100ms`、阈值 `1`、突触增益 `1.5`、恒定电流 `0.180`、1.2Hz Bernoulli 背景活动（幅值 `0.22`）。

神经活动到研究信号的映射：原 `DNa02/DNg13` 右转池作为 BUY 证据，左转池作为 SELL 证据，`score=(buy_rate-sell_rate)*1100`；绝对值小于 8 为 HOLD。原 `DNg100` 和 `DNp01/DNp10` 仅作为 arousal 记录，不直接决定方向。日志逐行保留原始行情、刺激特征、神经活动和最终解释。

这些映射是工程实验，不是生物学结论、投资建议或经过训练的策略。

## 准备 MaleCNS 缓存

本项目兼容 ornata/fly 生成的缓存格式。已下载的参考仓库位于被 Git 忽略的 `_reference/ornata-fly`：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH = "_reference/ornata-fly"
.\.venv\Scripts\python.exe -m fly64.data --prepare --cache data/malecns
```

官方原始数据约需 1.1GB，准备过程会产生 `model.npz`、`weights.npz` 和 `manifest.json`。`data/` 已被忽略，不会提交。

## 配置并运行 IEX

在 Alpaca 创建免费的 Market Data API key，然后只设置当前终端的环境变量；不要把真实值写入仓库：

```powershell
$env:APCA_API_KEY_ID = "你的 key id"
$env:APCA_API_SECRET_KEY = "你的 secret"
.\.venv\Scripts\fly-trader.exe --source iex
```

启动后会自动打开本地仪表盘 `http://127.0.0.1:8787/`。页面显示最新价格、行情源、报价年龄、原始/平滑神经分数、稳定信号和滚动曲线，并实时比较 fly_raw、fly_filtered、risk_executable、买入持有及断开连接组的本次进程假想收益、回撤和交易次数。收益口径默认计入 3bp 手续费与 5bp 滑点，不代表账户真实收益。页面可暂停或继续实验；服务只绑定本机，不展示 API 密钥，也没有下单按钮。

限定测试时长可使用 `--duration 60`。默认持续运行，按 `Ctrl+C` 停止。使用 `--no-browser` 可只启动服务而不自动打开页面，`--dashboard-port 8790` 可修改端口。凭据变量示例见 `.env.example`，`.env` 和 `.env.*` 均已被忽略。

## 老虎延时行情兼容入口

配置文件默认使用 `config/tiger_openapi_config.properties`，该文件已被 `.gitignore` 排除：

```powershell
.\.venv\Scripts\fly-trader.exe --source tiger --symbols AAPL --once
```

信号写入 `logs/signals.jsonl`。终端只打印证券代码、行情时间、信号与分数，不打印配置或凭据。

## 长桥港股只读行情

港股入口只创建长桥行情上下文，不创建交易上下文，也没有下单方法。程序启动时查询当前 OpenAPI 行情等级：若实际权限为 `BMP`，使用约 15 分钟延迟的定时拉取，并在终端、Web 页面和 JSONL 的 `feed` 中明确标记；若账户具备港股 OpenAPI 实时权限，则订阅报价推送。订阅被服务端拒绝时自动降级为轮询，不会把延迟数据标成实时行情。

在长桥开发者中心创建 API Key 后，可把密钥填入仓库根目录已被 Git 忽略的 `.env` 文件。程序启动时会自动加载，并以该文件覆盖当前 PowerShell 中可能残留的旧值。不要把真实值写入 `.env.example` 或其他配置文件：

```powershell
$env:LONGBRIDGE_APP_KEY = "你的 App Key"
$env:LONGBRIDGE_APP_SECRET = "你的 App Secret"
$env:LONGBRIDGE_ACCESS_TOKEN = "你的 Access Token"
.\.venv\Scripts\fly-trader.exe --source longbridge --symbols 700.HK 9988.HK 3690.HK
```

港股代码使用长桥的 `ticker.HK` 格式；纯数字输入也会自动转换，例如 `00700` 转为 `700.HK`。港股模式同样打开本地仪表盘，日志模式为 `longbridge-hk-signal-only-no-orders`。

## 美股与港股同时监听

默认 `--source both` 同时启动 Alpaca 美股和长桥港股只读行情，共享一份 MaleCNS 连接矩阵、每只股票保持独立神经状态。Web 页面提供“港股 / 美股”标签；港股连续交易时段默认显示港股，美国常规交易时段默认显示美股，也可随时手动切换。

在 Windows 资源管理器中也可以直接双击项目根目录的 `启动 Fly Trader.cmd`。启动器会自动切换到项目根目录并执行双市场模式；程序停止或启动失败后窗口会保留，以便查看输出。

```powershell
.\.venv\Scripts\fly-trader.exe --source both
```

默认美股为英伟达，默认港股为腾讯和智谱。可分别用 `--symbols` 和 `--hk-symbols` 修改。双市场日志分开保存为 `logs/signals-us.jsonl` 与 `logs/signals-hk.jsonl`，避免并发写入同一文件。

页面每 15 秒只读刷新同一个长桥 OpenAPI 模拟账户，展示净资产、现金、购买力和持仓；港股与美股标签分别过滤 `.HK` 和美股持仓。Alpaca 仅提供美股行情，不再作为账户监控来源。账户模块没有下单、改单或撤单函数；网络失败时保留并明确标记上次成功数据，同时自动重试，不会影响行情或神经模型运行。

## 离线回放与评估

可对一个或多个历史 JSONL 文件进行只读评估：

    .\.venv\Scripts\fly-trader.exe --source replay --replay-log logs\signals-us.jsonl logs\signals-hk.jsonl --report logs\replay-report.json

报告按标的分别输出 fly_raw、fly_filtered、risk_executable，并与买入持有、随机同频和断开连接组期望值对照，包含收益、最大回撤、交易次数、胜率、换手率、暴露比例和数据缺口。默认只接纳 fly-core-v1 的 schema 8 数据，旧事件只有显式添加 --include-legacy 才会进入报告。默认假设手续费 3bp、滑点 5bp，可用 --fee-bps 和 --slippage-bps 调整。该报告是研究回放，不代表真实可成交收益。

## 运行可靠性与目录

- JSONL 达到 50MB 后自动轮转并保留五份历史文件，异常退出留下的不完整尾行会在下次写入前清理。
- logs/*.state.json 与对应状态目录每 30 秒保存神经动态、随机数、过滤器和风控状态，重启时自动恢复。
- 风控内置 2026–2027 美股与港股休市日；超出日历覆盖范围时拒绝生成提案。
- 根目录仅保留项目入口和标准目录：src/ 源码、tests/ 测试、docs/ 文档、config/ 配置、data/ 模型数据、logs/ 运行数据、_reference/ 上游参考以及 .venv/ 本地环境。测试临时文件使用系统临时目录。

## 来源

连接组缓存格式、细胞群和神经动力学来自 [ornata/fly](https://github.com/ornata/fly)；其 MaleCNS 数据来自 Janelia MaleCNS v1.0。没有迁移 Fly64 的 macOS 游戏、共享内存、SM64、浏览器仪表盘或录屏桥接。
