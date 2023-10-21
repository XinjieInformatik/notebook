

通过pytest mark机制, 提供了多种不同级别的并行粒度(并发测试用例数量是可以配置的，这里只是举例)
- mr_ci_medium_parallel: 并行运行的测试用例数量为4
- mr_ci_serial: 无并行
- daily_ci: 仅在Daily测试时运行, 并行运行的测试用例数量为8
- daily_ci_medium_parallel: 仅在Daily测试时运行, 并行运行的测试用例数量为4
- daily_ci_serial: 仅在Daily测试时运行, 无并行
- 其他(默认情况): 并行运行的测试用例数量为8



```sh
# mac install brew
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
```
