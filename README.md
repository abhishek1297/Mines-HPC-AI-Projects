# Vector Reduction with Pthread and AVX

## Output 1:
```
[INFO] Vector length: 1048576
[INFO] Running threads: 16
[INFO] Init success
[INFO] Warmup run finished
[INFO] Executing operations 10 times

Execution details:

operation       exec time (msec)        speed up        sum error
==========================================================================
scalar seq      27.615                  1.00            0.000000e+00
vector seq      21.457                  1.28            1.514902e-06
scalar thr      6.601                   4.19            0.000000e+00
vector thr      5.150                   5.30            1.833772e-04
```

## Output 2:
```
[INFO] Vector length: 104857600
[INFO] Running threads: 16
[INFO] Init success
[INFO] Warmup run finished
[INFO] Executing operations 10 times

Execution details:

operation       exec time (msec)        speed up        sum error
==========================================================================
scalar seq      2777.403                1.00            0.000000e+00
vector seq      2255.632                1.24            7.917623e-03
scalar thr      252.607                 12.66           0.000000e+00
vector thr      192.139                 16.69           4.272391e-05
```

## Output 3:
```
[INFO] Vector length: 10485760
[INFO] Running threads: 24
[INFO] Init success
[INFO] Warmup run finished
[INFO] Executing operations 10 times

Execution details:

operation       exec time (msec)        speed up        sum error
==========================================================================
scalar seq      1923.775                1.00            0.000000e+00
vector seq      1776.627                1.08            4.060561e-06
scalar thr      115.018                 16.43           0.000000e+00
vector thr      106.430                 18.36           1.413854e-05
```

## Output 4:
```
[INFO] Vector length: 104857600
[INFO] Running threads: 24
[INFO] Init success
[INFO] Warmup run finished
[INFO] Executing operations 10 times

Execution details:

operation       exec time (msec)        speed up        sum error
==========================================================================
scalar seq      19223.555               1.00            0.000000e+00
vector seq      17716.317               1.08            7.917622e-03
scalar thr      1064.356                18.77           0.000000e+00
vector thr      972.782                 20.54           2.212295e-06
```