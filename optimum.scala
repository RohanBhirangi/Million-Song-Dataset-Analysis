import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.abs
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{min, max, lit}
import org.apache.spark.sql.SQLContext
val sqlContext = new SQLContext(sc)

val usdf = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/PrData/user1.csv")

val msdf = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/PrData/MSDmain.csv")
var msdf2 = msdf.withColumn("time_signature", when(col("time_signature_confidence")<0.5, 4).otherwise(col("time_signature")))
var msdf3 = msdf2.select($"duration", $"loudness", $"artist_hotttnesss", $"tempo"/$"time_signature" as "speed", $"song_id")
var msdf4 = msdf3.na.drop()

// Without feature scaling
// var allDF = msdf4

// With feature scaling

val scaledRange = lit(4)
val scaledMin = lit(1)

@transient val (dMin, dMax) = msdf4.agg(min($"duration"), max($"duration")).first match { case Row(x: Double, y: Double) => (x,y)}
@transient val dNorm = ($"duration"-dMin)/(dMax-dMin)
@transient val dScal = scaledRange*dNorm + scaledMin

@transient val (lMin, lMax) = msdf4.agg(min($"loudness"), max($"loudness")).first match { case Row(x: Double, y: Double) => (x,y)}
@transient val lNorm = ($"loudness"-lMin)/(lMax-lMin)
@transient val lScal = scaledRange*lNorm + scaledMin

@transient val (aMin, aMax) = msdf4.agg(min($"artist_hotttnesss"), max($"artist_hotttnesss")).first match { case Row(x: Double, y: Double) => (x,y)}
@transient val aNorm = ($"artist_hotttnesss"-aMin)/(aMax-aMin)
@transient val aScal = scaledRange*aNorm + scaledMin

@transient val (sMin, sMax) = msdf4.agg(min($"speed"), max($"speed")).first match { case Row(x: Double, y: Double) => (x,y)}
@transient val sNorm = ($"speed"-sMin)/(sMax-sMin)
@transient val sScal = scaledRange*sNorm + scaledMin

val msdf5 = msdf4.withColumn("dScal",dScal).withColumn("lScal", lScal).withColumn("aScal", aScal).withColumn("sScal", sScal)
val allDF = msdf5.select("dScal", "lScal", "aScal", "sScal", "song_id")

val rowsRDD = allDF.rdd.map(r => (r.getDouble(0), r.getDouble(1), r.getDouble(2), r.getDouble(3), r.getString(4)))
rowsRDD.cache()
val vectors = allDF.rdd.map(r => Vectors.dense(r.getDouble(0), r.getDouble(1), r.getDouble(2), r.getDouble(3)))
vectors.cache()

for(k <- 2 to 20)
{
	val kMeansModel = KMeans.train(vectors, k, 20)
	val WSSE = kMeansModel.computeCost(vectors)
	println("K = " + k)
	println("Within set SSE = " + WSSE)
}
