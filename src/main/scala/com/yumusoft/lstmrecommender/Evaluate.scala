package com.yumusoft.lstmrecommender

import java.io.File

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.util.ModelSerializer
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class EvaluateConfig(
  input: File = null,
  modelName: String = ""
)

object EvaluateConfig {
  val parser = new OptionParser[EvaluateConfig]("Evaluate") {
      head("lstmrecommender Evaluate", "1.0")

      opt[File]('i', "input")
        .required()
        .valueName("<file>")
        .action( (x, c) => c.copy(input = x) )
        .text("The file with test data.")

      opt[String]('m', "model")
        .required()
        .valueName("<modelName>")
        .action( (x, c) => c.copy(modelName = x) )
        .text("Name of trained model file.")
    }

    def parse(args: Array[String]): Option[EvaluateConfig] = {
      parser.parse(args, EvaluateConfig())
    }
}

object Evaluate {
  private val log = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    EvaluateConfig.parse(args) match {
      case Some(config) =>
        val model = ModelSerializer.restoreMultiLayerNetwork(config.modelName)
        val (testData, normalizer) = DataIterators.irisCsv(config.input)
        normalizer.load((1 to 4).map(j => new File(config.modelName + s".norm$j")):_*)

        val eval = new Evaluation(3)
        while (testData.hasNext) {
            val ds = testData.next()
            val output = model.output(ds.getFeatureMatrix)
            eval.eval(ds.getLabels, output)
        }

        log.info(eval.stats())

      case _ =>
        log.error("Invalid arguments.")
    }
  }
}
