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

// Get the best k 
for(k <- 2 to 20)
{
	val kMeansModel = KMeans.train(vectors, k, 20)
	val WSSE = kMeansModel.computeCost(vectors)
	println("K = " + k)
	println("Within set SSE = " + WSSE)
}

val kMeansModel = KMeans.train(vectors, 7, 20)
kMeansModel.clusterCenters.foreach(println)

val predictions = rowsRDD.map{r => (r._5, kMeansModel.predict(Vectors.dense(r._1, r._2, r._3, r._4)))}
//val predictions = allDF.map{r => (r.getString(4), kMeansModel.predict(Vectors.dense(r.getDouble(0), r.getDouble(1), r.getDouble(2), r.getDouble(3))))}
val songclusters = predictions.toDF("SongID", "Cluster")
//songclusters.show()

val userclusters = usdf.join(songclusters, usdf("SongId")===songclusters("SongID"))
val topfeats = userclusters.join(allDF, userclusters("SongId")===allDF("song_id")).orderBy(desc("PlayCount")).select("Cluster","duration","loudness","artist_hotttnesss","speed")
topfeats.show()
val tf = topfeats.head(3)

val song1 = tf(0)
val song2 = tf(1)
val song3 = tf(2)

val msdclusters = allDF.join(songclusters, allDF("song_id")===songclusters("SongID"))
val cluster1 = msdclusters.filter(msdclusters("Cluster")===song1(0))
val cluster2 = msdclusters.filter(msdclusters("Cluster")===song2(0))
val cluster3 = msdclusters.filter(msdclusters("Cluster")===song3(0))
cluster1.show()
cluster2.show()
cluster3.show()

val pred1 = cluster1.select($"song_id", abs($"duration"-song1(1)) + abs($"loudness"-song1(2)) + abs($"artist_hotttnesss"-song1(3)) + abs($"speed"-song1(4)) as "score").orderBy(asc("score")).limit(4)
val pred2 = cluster2.select($"song_id", abs($"duration"-song2(1)) + abs($"loudness"-song2(2)) + abs($"artist_hotttnesss"-song2(3)) + abs($"speed"-song2(4)) as "score").orderBy(asc("score")).limit(4)
val pred3 = cluster3.select($"song_id", abs($"duration"-song3(1)) + abs($"loudness"-song3(2)) + abs($"artist_hotttnesss"-song3(3)) + abs($"speed"-song3(4)) as "score").orderBy(asc("score")).limit(4)

pred1.join(msdf, pred1("song_id")===msdf("song_id")).select("score", "title").show()
pred2.join(msdf, pred2("song_id")===msdf("song_id")).select("score", "title").show()
pred3.join(msdf, pred3("song_id")===msdf("song_id")).select("score", "title").show()

System.exit(1)
