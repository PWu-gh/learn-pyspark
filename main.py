import time
import random
from functools import reduce
import math

try:
    from pyspark.sql import SparkSession
    from pyspark import SparkContext
except ImportError:
    print("PySpark is not installed. Only standard Python benchmark will be run.")

def point_in_circle(x, y):
    return x*x + y*y <= 1

def monte_carlo_pi(n):
    inside_circle = sum(point_in_circle(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n))
    return 4 * inside_circle / n

def benchmark_standard_python(iterations, points_per_iteration):
    start_time = time.time()
    for _ in range(iterations):
        pi_estimate = monte_carlo_pi(points_per_iteration)
    end_time = time.time()
    return end_time - start_time, pi_estimate

def monte_carlo_pi_spark(spark, n):
    def generate_and_check_point(_):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        return 1 if x*x + y*y <= 1 else 0

    count = spark.sparkContext.parallelize(range(n), 100).map(generate_and_check_point).sum()
    return 4.0 * count / n

def benchmark_pyspark(spark, iterations, points_per_iteration):
    start_time = time.time()
    for _ in range(iterations):
        pi_estimate = monte_carlo_pi_spark(spark, points_per_iteration)
    end_time = time.time()
    return end_time - start_time, pi_estimate

def main():
    iterations = 10
    points_per_iteration = 100_000_000  # Increase this for more accuracy and longer runtime

    # Standard Python benchmark
    std_time, std_pi = benchmark_standard_python(iterations, points_per_iteration)
    print(f"Standard Python:")
    print(f"  Total time: {std_time:.2f} seconds")
    print(f"  Average time per iteration: {std_time / iterations:.2f} seconds")
    print(f"  Final Pi estimate: {std_pi}")
    print(f"  Error: {abs(std_pi - math.pi):.6f}")

    # PySpark benchmark
    try:
        spark = SparkSession.builder.appName("MonteCarloPi").getOrCreate()
        spark_time, spark_pi = benchmark_pyspark(spark, iterations, points_per_iteration)
        print(f"\nPySpark:")
        print(f"  Total time: {spark_time:.2f} seconds")
        print(f"  Average time per iteration: {spark_time / iterations:.2f} seconds")
        print(f"  Final Pi estimate: {spark_pi}")
        print(f"  Error: {abs(spark_pi - math.pi):.6f}")
        spark.stop()
    except NameError:
        print("\nPySpark benchmark skipped due to missing PySpark installation.")

if __name__ == "__main__":
    main()