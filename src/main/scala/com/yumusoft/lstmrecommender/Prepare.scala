package com.yumusoft.lstmrecommender

import java.io._
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.text.SimpleDateFormat
import java.util.{Calendar, GregorianCalendar}

import com.yumusoft.lstmrecommender.Train.getClass
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.InputStreamInputSplit
import org.slf4j.LoggerFactory
import scopt.OptionParser

import scala.collection.immutable.{HashMap, TreeSet}
import scala.collection.JavaConverters._


case class PrepareConfig(
  input: File = null,
  outputDir: String = "",
  count: Int = 0,
  length: Int = 32
)

object PrepareConfig {
  val parser = new OptionParser[PrepareConfig]("Prepare") {
    head("lstmrecommender Prepare", "1.0")

    opt[File]('i', "input")
      .required()
      .valueName("<file>")
      .action( (x, c) => c.copy(input = x) )
      .text("The file with the raw Online Retail data.")

    opt[String]('o', "outputDir")
      .required()
      .valueName("<outputDir>")
      .action( (x, c) => c.copy(outputDir = x) )
      .text("The directory where sequence CSVs are stored.")

    opt[Int]('c', "count")
      .action( (x, c) => c.copy(count = x) )
      .text("Optional number of sequences to output.  Default is whole file.")

    opt[Int]('l', "length")
      .action( (x, c) => c.copy(length = x) )
      .text("Length of sequences. (All files will be padded to this length.)")
  }

  def parse(args: Array[String]): Option[PrepareConfig] = {
    parser.parse(args, PrepareConfig())
  }
}

object Prepare {
  private val log = LoggerFactory.getLogger(getClass)

  private val dateParser = new SimpleDateFormat("M/d/yyyy HH:mm")

  def dateIds(dateStr: String): (Int, Int) = {
    val cal = Calendar.getInstance()
    val d = dateParser.parse(dateStr)
    cal.setTime(d)

    (cal.get(Calendar.MONTH), cal.get(Calendar.DAY_OF_WEEK))
  }

  def main(args: Array[String]): Unit = {
    PrepareConfig.parse(args) match {
      case Some(config) =>
        if (new File(config.outputDir).exists()) {
          log.error("Output Directory already exists.")
        } else {
          log.info("Processing file")

          val rr = new CSVRecordReader(1, ",")
          rr.initialize(new InputStreamInputSplit(new FileInputStream(config.input)))

          var itemCounts = new HashMap[String, Int]()

          var countrySet = new HashMap[String, Int]()
          var nextCountryId = 1

          while (rr.hasNext) {
            val row = rr.next()
            if (row.size() == 8) {
              val (itemDesc, country) = (row.get(2).toString, row.get(7).toString)

              if (!countrySet.contains(country)) {
                countrySet += (country -> nextCountryId)
                nextCountryId += 1
              }

              val item = itemDesc.trim.replace(",", "").replace("\"", "").replace("'", "")
              if (item.length > 0 && item.length < 100) {
                if (itemCounts.contains(item)) {
                  itemCounts += item -> (itemCounts(item) + 1)
                } else {
                  itemCounts += item -> 1
                }
              }
            }
          }

          val itemSet = itemCounts
            .toSeq
            .filter { case (_, c) => c >= 100 && c < 1000}
            .sortBy { case (_, c) => c }
            .zipWithIndex
            .map { case ((item, c), i) => (item, i) }
            .toMap

          rr.reset()
          rr.initialize(new InputStreamInputSplit(new FileInputStream(config.input)))

          new File(config.outputDir + "/sessions/").mkdirs()

          var rows: List[(String, String, String)] = Nil
          var count = 0
          var sessionId = ""

          while (rr.hasNext && count < config.count) {
            val row = rr.next()
            if (row.size() == 8) {
              val Seq(currentSession, itemDesc, date, country) = Seq(0, 2, 4, 7).map(row.get(_).toString)

              if (currentSession != sessionId) {
                if (rows.size > 5 && rows.size < config.length) {
                  count += 1
                  log.info(s"Writing session $count with ${rows.size} items.")

                  val rowIds = rows.reverse.map { case (i, c, d) => (itemSet(i), countrySet(c), dateIds(d)) }

                  val currentInputFile = new File(config.outputDir + "/sessions/Input_" + count + ".csv")
                  currentInputFile.createNewFile()
                  val currentLabelFile = new File(config.outputDir + "/sessions/Label_" + count + ".csv")
                  currentLabelFile.createNewFile()

                  val inputLines = rowIds.sliding(2).map { case List((i, c, (m, w)), _) => s"$i,$c,$m,$w" }
                  val labelLines = rowIds.sliding(2).map { case List(_, (next, _, (_, _))) => s"$next" }

                  Files.write(
                    Paths.get(currentInputFile.getAbsolutePath),
                    inputLines.mkString("\n").getBytes()
                  )

                  Files.write(
                    Paths.get(currentLabelFile.getAbsolutePath),
                    labelLines.mkString("\n").getBytes()
                  )
                }

                rows = Nil
                sessionId = row.get(0).toString
              }

              val item = itemDesc.trim.replace(",", "").replace("\"", "").replace("'", "")
              if (itemSet.contains(item)) {
                rows ::= (item, country, date)
              }
            }
          }

          log.info("Writing items dictionary")

          val itemFile = new File(config.outputDir + "/items.csv")
          val itemOut = new PrintWriter(new BufferedWriter(new FileWriter(itemFile, true)))
          itemSet.toSeq.sortBy { case (_, itemId) => itemId }.foreach { case (desc, index) =>
            itemOut.println(s"$desc,$index")
          }
          itemOut.close()

          val countryFile = new File(config.outputDir + "/countries.csv")
          val countryOut = new PrintWriter(new BufferedWriter(new FileWriter(countryFile, true)))
          countrySet.toSeq.sortBy { case (_, countryId) => countryId }.foreach { case (country, index) =>
            countryOut.println(s"$country,$index")
          }
          countryOut.close()

          log.info("Processing finished.")
        }
      case _ =>
        log.error("Invalid arguments.")
    }
  }
}
