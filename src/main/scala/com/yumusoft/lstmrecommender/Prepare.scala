package com.yumusoft.lstmrecommender

import java.io._
import java.nio.file.{Files, Paths, StandardOpenOption}

import com.yumusoft.lstmrecommender.Train.getClass
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.InputStreamInputSplit
import org.slf4j.LoggerFactory
import scopt.OptionParser

import scala.collection.immutable.{HashMap, TreeSet}


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

  private val START = "<start>"
  private val END = "<end>"

  def main(args: Array[String]): Unit = {
    PrepareConfig.parse(args) match {
      case Some(config) =>
        if (new File(config.outputDir).exists()) {
          log.error("Output Directory already exists.")
        } else {
          log.info("Processing file")

          val rr = new CSVRecordReader(1, ",")
          rr.initialize(new InputStreamInputSplit(new FileInputStream(config.input)))

          var itemSet = new HashMap[String, Int]()
          //itemSet += (START -> 0)
          //itemSet += (END -> 0)
          var nextItemId = 1

          var countrySet = new HashMap[String, Int]()
          var nextCountryId = 1

          var sessionId: String = ""
          var items: List[String] = Nil
          while (rr.hasNext) {
            val row = rr.next()
            if (row.get(0).toString != sessionId) {
              if (items.size > 1 && items.size <= config.length) {
                for (item <- items.reverse) {
                  if (!itemSet.contains(item)) {
                    itemSet += (item -> nextItemId)
                    nextItemId += 1
                  }


                  val country = row.get(row.size() - 1).toString
                  if (!countrySet.contains(country)) {
                    countrySet += (country -> nextCountryId)
                    nextCountryId += 1
                  }
                }
              }

              items = Nil
              sessionId = row.get(0).toString
            }

            val item = row.get(2).toString.trim.replace(",", "").replace("\"", "").replace("'", "")
            if (item.length >  0 && item.length < 100) {
              items ::= item
            }
          }

          rr.reset()
          rr.initialize(new InputStreamInputSplit(new FileInputStream(config.input)))

          new File(config.outputDir + "/sessions/").mkdirs()

          var rows: List[(String, String)] = Nil

          var count = 0
          while (rr.hasNext && count < config.count) {
            val row = rr.next()
            if (row.get(0).toString != sessionId) {
              if (rows.size > 1 && rows.size < config.length) {
                count += 1
                log.info(s"Writing session $count with ${rows.size} items.")

                val rowIds = rows.reverse.map { case (i, c) => (itemSet(i), countrySet(c)) }

                val currentInputFile = new File(config.outputDir + "/sessions/Input_" + count + ".csv")
                currentInputFile.createNewFile()
                val currentLabelFile = new File(config.outputDir + "/sessions/Label_" + count + ".csv")
                currentLabelFile.createNewFile()


                val inputLines = rowIds.sliding(2).map { case List(a, b) => s"${a._1},${a._2}" }
                val labelLines = rowIds.sliding(2).map { case List(a, b) => s"${b._1}" }

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

            val item = row.get(2).toString.trim.replace(",", "").replace("\"", "").replace("'", "")
            val country = row.get(row.size() - 1).toString
            if (item.length > 0 && item.length < 100) {
              rows ::= (item, country)
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
