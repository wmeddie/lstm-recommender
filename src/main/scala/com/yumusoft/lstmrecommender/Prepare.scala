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
  count: Int = 0
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
  }

  def parse(args: Array[String]): Option[PrepareConfig] = {
    parser.parse(args, PrepareConfig())
  }
}

object Prepare {
  private val log = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    PrepareConfig.parse(args) match {
      case Some(config) =>
        if (new File(config.outputDir).exists()) {
          log.error("Output Directory already exists.")
        } else {
          log.info("Processing file")

          val rr = new CSVRecordReader()
          rr.initialize(new InputStreamInputSplit(new FileInputStream(config.input)))

          var itemSet = new HashMap[String, Int]()
          var lastItemId = 0

          var sessionId: String = ""
          var items: List[String] = Nil
          while (rr.hasNext) {
            val row = rr.next()
            if (row.get(0).toString != sessionId) {
              if (items.size > 1 && items.size <= 30) {
                var itemIds: List[Int] = Nil
                for (item <- items) {
                  if (!itemSet.contains(item)) {
                    itemSet = itemSet + (item -> lastItemId)
                    lastItemId += 1
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

          rr.initialize(new InputStreamInputSplit(new FileInputStream(config.input)))

          new File(config.outputDir + "/sessions/").mkdirs()

          items = Nil
          var count = 0
          while (rr.hasNext) {
            val row = rr.next()
            if (row.get(0).toString != sessionId) {
              if (items.size > 1 && items.size <= 30) {
                count += 1
                log.info(s"Writing session $count with ${items.size} items.")

                val itemIds: List[Int] = items.map(i => itemSet(i))

                val currentFile = new File(config.outputDir + "/sessions/" + sessionId + ".csv")
                if (!currentFile.exists()) {
                  currentFile.createNewFile()
                }

                val lines = itemIds.reverse.sliding(2).map { case List(a, b) =>
                  val line = Array.ofDim[Int](itemSet.size + 1)
                  line(itemSet.size) = b
                  line(a) = 1
                  line.mkString(",")
                }

                Files.write(
                  Paths.get(currentFile.getAbsolutePath),
                  lines.mkString("\n").getBytes(),
                  StandardOpenOption.APPEND
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
