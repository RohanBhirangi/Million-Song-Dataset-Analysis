import org.apache.spark.sql.SQLContext
val sqlContext = new SQLContext(sc)
val lydf = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/PrData/lyrics.csv")
val gndf = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/PrData/genre.csv")
val u1df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/PrData/user1.csv")
val msdf = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/PrData/MSDmain.csv")
val stdf = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("file:/PrData/style.csv")

var stdf2 = stdf.select("style", "track_id")
var gndf2 = gndf.select("genre", "track_id")
var lydf2 = lydf.select("track_id", "word_count")
var u1df2 = u1df.select("SongId", "PlayCount")
var msdf2 = msdf.select("track_id", "song_id",  "duration", "key", "loudness", "mode", "tempo", "time_signature", "year", "artist_name", "title")

var msdgnr = gndf2.join(msdf2, gndf2("track_id")===msdf2("track_id"))
var lyrgnr = gndf2.join(lydf2, gndf2("track_id")===lydf2("track_id"))
var usrmsd = u1df2.join(msdf3, u1df2("SongId")===msdf3("song_id"))

var gnrusr = usrmsd.join(gndf2, usrmsd("track_id")===gndf2("track_id"))
var usrfavs = gnrusr.select("PlayCount", "genre")
var gp = usrfavs.groupBy("genre").agg(sum("PlayCount"))
gp.show

var gnryear = msdgnr.groupBy("year", "genre").count().orderBy(desc("year"))

var msdf3 = msdf.withColumn("time_signature", when(col("time_signature_confidence")<0.5, 4).otherwise(col("time_signature")))
var features = msdf3.select($"track_id", $"title", $"duration", $"loudness", $"artist_hotttnesss", $"tempo"/$"time_signature" as "speed")

var g = msdf.select("track_id", "artist_latitude", "artist_longitude")
var geospatgenre2 = gndf2.join(g, gndf2("track_id")===g("track_id"))
g.write.format("com.databricks.spark.csv").option("header","true").save("file:/genrelocmap")

