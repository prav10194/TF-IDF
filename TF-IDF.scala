import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.commons.io.FileUtils;
import java.io.{File,FileInputStream,FileOutputStream}
import scala.collection.mutable.ListBuffer

/**
 * Run code using the following commands -
 *
 * 1. val args = Array("datasets","test.txt")
 *    datasets is the folder containg all the datasets and test.txt is the file containg the words on which we will run our code
 *
 * 2. :load TF-IDF.scala
 *
 * 3. SimpleApp.main(args)
 */


object SimpleApp {
  def main(args: Array[String]) {

val training=args(0)
val testingfile=args(1)


/**
 * termfreq reads all the files at once and outputs Array[(String, String)].
 * For same word in different files: it will look something like - Array((kv07,(alt_atheism-eaeac.txt,4), (talk_religion_misc-8faa1.txt,3))
 * Here kv07 is the word, alt_atheism-eaeac.txt is the file and 4 is the tf of the word in that file. Same for the next one too.
 */

 /**
  * I am splitting the file path on line 47 and 75 using "/" and taking the last. 
  * So for example in my case the path looks like - /Users/folder1/folder2/datasets/comp.graphics.txt
  * So I am taking the last value after split which will give me the filename.
  * In case of other OS, the split might be on a different character.
  */

val termfreq = sc.wholeTextFiles(training).flatMap {
      case (path, text) =>
        text.split("""\W+""")
          .map {
            word => (word, path)
          }
    }.map {
      case (w, p) => ((w, p.split("/").last), 1)
    }.reduceByKey {
      case (n1, n2) => n1 + n2
    }.map {
      case ((w, p), n) => (w, (p, n))
    }.groupBy {
      case (w, (p, n)) => w
    }.map {
      case (w, seq) =>
        val seq2 = seq map {
          case (_, (p, n)) => (p, n)
        }
        (w, seq2.mkString(", "))
    }

/**
 * tempMap stores the file in the form of Array[((String, String), Int)].
 * Eg - Array(((injury,rec_motorcycles-bf2ba.txt),1)
 * We will use it to calculate df
 */

val tempMap=sc.wholeTextFiles(training).flatMap {
      case (path, text) =>
        text.split("""\W+""")
          .map {
            word => (word, path)
          }
    }.map {
      case (w, p) => ((w, p.split("/").last), 1)
    }.reduceByKey {
      case (n1, n2) => n1 + n2
    }

/**
 * df converts the tempMap to - Array[(String, Int)].
 * Eg - Array((kv07,2))
 * Here 2 is the number of unique documents in which we found the word kv07. This is the df count of the word.
 */

val df=tempMap.map{
  x=>(x._1)
}.map
{
  x=>(x._1,1)
}
.reduceByKey(_+_)
df.collect()

/**
 * tfidf joins termfreq with df. RDD is in the form of Array[(String, (String, Int))] = Array((kv07,((alt_atheism-eaeac.txt,4), (talk_religion_misc-8faa1.txt,3),2)).
 * Here kv07 is the word, alt_atheism-eaeac.txt is the text file, 4 is tf and 2 at the end is the df of the word, since its occurring in 2 files.
 */

val tfidf = termfreq.join(df)


/**
 * final_weight will have the tf-idf calculated for each word. It contains a collection of ListBuffers in the form - Array[Array[scala.collection.mutable.ListBuffer[String]]].
 * Eg - For word kv07 - there are 2 List Buffers created, one for each file.
 * ListBuffer(kv07, alt_atheism-eaeac.txt, 9.210340371976184)
 * ListBuffer(kv07, talk_religion_misc-8faa1.txt, 6.907755278982138)
 */


val final_weight = for(a <- tfidf)
yield {
  val word=a._1
  val docf=a._2._2
  val values = a._2._1
  val rddvalues = values.split(" ")

  val word_tuple_final = for(b <- rddvalues)
  yield {
    val w = b.split(",")(1).replace("(","").replace(")","")
    val docpath = b.split(",")(0).replace("(","").replace(")","")
    var weight=w.toInt*scala.math.log(20/docf.toInt)

    var word_tuple = new ListBuffer[String]()
    word_tuple += word
    word_tuple += docpath
    word_tuple += weight.toString
    word_tuple
  }
  word_tuple_final
}

/**
 * tfidf_final is sorting the final_weight according to the weight calculated.
 * Key will be the the word and Value will be the combination of (filename, weight)
 * Eg - (Denning,(sci_crypt-ce7a9.txt,125.82075548926761))
 */

val tfidf_final = final_weight.flatMap(x=>x).collect().map(x=>(x(0),(x(1),x(2)))).sortBy{case (word: String, (documentId: String, tfidfvalue: String)) => -tfidfvalue.toDouble}

/**
 * test_file reads the words from our test file and test_words converts it to a Set of words split on space.
 * Array[String] = Array(science ,nothin, baseball, America)
 */

val test_file = sc.textFile(testingfile)
val test_words=test_file.flatMap(x=>x.split(" ")).filter(x=>x.length!=0)

/**
 * Read each word in test_words and use it to filter our tfidf_final containing all the possible combinations.
 * After that sort it in descending order and take the top 5.
 */

val document_weights = for(tw<-test_words) yield{
  tfidf_final.filter(x=>x._1 == tw).sortBy{case (word: String, (documentId: String, tfidfvalue: String)) => -tfidfvalue.toDouble}.take(5)
}

/**
 * Print the document_weights line by line.
 */

document_weights.collect.foreach(indvArray => indvArray.foreach(println))

  }
}
