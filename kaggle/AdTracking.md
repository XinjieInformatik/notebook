# AdTracking classification

## 原始数据

Each row of the training data contains a click record, with the following features.

`ip`: 点击的ip地址
`app`: app id
`device`: 设备类型id号 (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
`os`: 手机操作系统id
`channel`: 广告通道id
`click_time`: 点击的时间戳 (UTC)
`attributed_time`: 点击后下载app的时间
`is_attributed`: target, app是否被下载

ps: Note that ip, app, device, os, and channel are encoded.

- ip: ip address of click.
- app: app id for marketing.
- device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
- os: os version id of user mobile phone
- channel: channel id of mobile ad publisher
- click_time: timestamp of click (UTC)
- attributed_time: if user download the app for after clicking an ad, this is the time of the app download
- is_attributed: the target that is to be predicted, indicating the app was downloaded
