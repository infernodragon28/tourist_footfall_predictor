from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, to_date
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("Tourist_Footfall_Prediction").getOrCreate()

# Load dataset
df = spark.read.csv("tourism_data.csv", header=True, inferSchema=True)
df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))

# Preprocessing
df = df.withColumn("Year", year("Date")) \
       .withColumn("Month", month("Date")) \
       .withColumn("Day", dayofmonth("Date")) \
       .dropna(subset=["WeatherIndex", "HolidayFlag", "StateGDP", "TouristCount"])

# Feature engineering
feature_cols = ["Year", "Month", "Day", "WeatherIndex", "HolidayFlag", "StateGDP"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train model
rf = RandomForestRegressor(featuresCol="features", labelCol="TouristCount", numTrees=100)
model = rf.fit(train_data)

# Evaluate
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="TouristCount", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Convert predictions to Pandas DataFrame
pdf = predictions.select("prediction", "TouristCount").toPandas()
# Convert full DataFrame to Pandas for extra plots
full_pdf = df.select("Year", "Month", "TouristCount", "WeatherIndex").toPandas()

# Histogram of Tourist Count
plt.figure(figsize=(10, 6))
sns.histplot(full_pdf["TouristCount"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Tourist Count")
plt.xlabel("Tourist Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram_tourist_count.png")

# Bar Plot: Average Tourist Count per Month
monthly_avg = full_pdf.groupby("Month")["TouristCount"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x="Month", y="TouristCount", data=monthly_avg, palette="viridis")
plt.title("Average Tourist Count per Month")
plt.xlabel("Month")
plt.ylabel("Average Tourist Count")
plt.tight_layout()
plt.savefig("barplot_avg_monthly_tourists.png")

# Plot: Actual vs Predicted Tourist Count
plt.figure(figsize=(10, 6))
sns.scatterplot(x="TouristCount", y="prediction", data=pdf, alpha=0.6)

# Add perfect prediction line
min_val = min(pdf["TouristCount"].min(), pdf["prediction"].min())
max_val = max(pdf["TouristCount"].max(), pdf["prediction"].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect Prediction")

# Labels and title
plt.title("Actual vs Predicted Tourist Count")
plt.xlabel("Actual Tourist Count")
plt.ylabel("Predicted Tourist Count")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.savefig("predicted_vs_actual.png")



# Save model
model.write().overwrite().save("tourist_footfall_model")


# Stop Spark
spark.stop()
