# sql 小站

## sql 语法

### row_number() over ()
```sql
row_number() over (partition by user_id order by time desc)
```
有一张订单表ordr_list，共有三列：ordr_id（订单id）-- bigint，uid（乘客id）-- bigint，start_time（发单时间）-- datetime，对于同一个乘客，每个订单的发单时间均不同.
求得每个用户最早发单的订单id.
```sql
select
user_id,
order_id
from (
  select
  user_id,
  order_id,
  row_number() over(partition by user_id order by start_time) as rk
  from order_table
  where partition_date = '$now.date'
)
where rk = 1
```

### DATEADD
DATEADD(单位, number, date)


## 知识点
### where 与 having 区别
1. 顺序: where 在 group by 前, having 在 group by 后
2. 聚合函数 having 中可以使用聚合函数进行过滤

```sql
select user_id, count(*) as cnt
from xx
group by user_id
having cnt > 3 (having count(*) > 3)

select user_id, cnt
from (
select user_id, count(*) as cnt
from xx
group by user_id
)
where cnt > 3
```


## sql 小练习
有一张订单表table1，表中包含user_id（用户id），dt（购买日期），amt（购买金额），找出购买天数和购买金额最多的用户，按要求输出用户id、购买天数、购买金额、备注是购买天数最多还是购买金额最多。
```sql
select
user_id, day_cnt, amt_sum
from (
  select
  user_id,
  count(distinct dt) as day_cnt,
  sum(amt) as amt_sum
  from table1
  group by 1
)
from table1
```

用SQL找出所有的连续3天登录用户
+──────────+─────────────+
| user_id  | login_date  |
+──────────+─────────────+
| A        | 2019-09-02  |
| A        | 2019-09-03  |
| A        | 2019-09-04  |
| B        | 2018-11-25  |

```sql
select user_id
from (
  select user_id, login_date, row_number() over(partition by user_id order by login_date) as rk
  from table
) t
group by user_id, DATEADD(D, -t.rk, login_date)
having count(user_id) >= 3
```
