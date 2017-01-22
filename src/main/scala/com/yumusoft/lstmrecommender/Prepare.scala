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
          itemSet += (END -> 0)
          var nextItemId = 1

          var sessionId: String = ""
          var items: List[String] = Nil
          while (rr.hasNext) {
            val row = rr.next()
            if (row.get(0).toString != sessionId) {
              if (items.size > 1 && items.size <= config.length) {
                var itemIds: List[Int] = Nil
                for (item <- items.reverse) {
                  if (!itemSet.contains(item)) {
                    itemSet = itemSet + (item -> nextItemId)
                    nextItemId += 1
                  }
                  itemIds = itemSet(item) :: itemIds
                }
              }

              items = Nil
              sessionId = row.get(0).toString
            }

            val item = row.get(2).toString.trim.replace(",", "").replace("\"", "").replace("'", "")
            if (item.length >  0 && item.length < 100) {
              items = item :: items
            }
          }

          rr.reset()
          rr.initialize(new InputStreamInputSplit(new FileInputStream(config.input)))

          new File(config.outputDir + "/sessions/").mkdirs()

          items = Nil
          var count = 0
          while (rr.hasNext && count < config.count) {
            val row = rr.next()
            if (row.get(0).toString != sessionId) {
              if (items.size > 1 && items.size < config.length) {
                count += 1
                log.info(s"Writing session $count with ${items.size} items.")

                items = items.reverse.padTo(config.length, END)
                val itemIds: List[Int] = items.map(i => itemSet(i))

                val currentInputFile = new File(config.outputDir + "/sessions/Input_" + count + ".csv")
                currentInputFile.createNewFile()
                val currentLabelFile = new File(config.outputDir + "/sessions/Label_" + count + ".csv")
                currentLabelFile.createNewFile()


                val inputLines = itemIds.sliding(2).map { case List(a, b) =>
                  //val line = Array.ofDim[Int](itemSet.size)
                  //line(a) = 1
                  //line.mkString(",")
                  s"$a"
                }

                val labelLines = itemIds.sliding(2).map { case List(a, b) =>
                  s"$b"
                }

                Files.write(
                  Paths.get(currentInputFile.getAbsolutePath),
                  inputLines.mkString("\n").getBytes()
                )

                Files.write(
                  Paths.get(currentLabelFile.getAbsolutePath),
                  labelLines.mkString("\n").getBytes()
                )
              }

              items = Nil
              sessionId = row.get(0).toString
            }

            val item = row.get(2).toString.trim.replace(",", "").replace("\"", "").replace("'", "")
            if (item.length > 0 && item.length < 100) {
              items = item :: items
            }
          }

          log.info("Writing items dictionary")

          val itemFile = new File(config.outputDir + "/items.csv")
          val itemOut = new PrintWriter(new BufferedWriter(new FileWriter(itemFile, true)))
          itemSet.toSeq.sortBy { case (_, itemId) => itemId }.foreach { case (desc, index) =>
            itemOut.println(s"$desc,$index")
          }
          itemOut.close()

          log.info("Processing finished.")
        }
      case _ =>
        log.error("Invalid arguments.")
    }
  }
}
