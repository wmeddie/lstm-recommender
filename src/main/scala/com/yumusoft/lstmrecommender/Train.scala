package com.yumusoft.lstmrecommender

import java.io.File
import java.nio.file.{Files, Paths}

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.api.layers.IOutputLayer
import org.deeplearning4j.nn.conf.graph.rnn.{DuplicateToTimeSeriesVertex, LastTimeStepVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class TrainConfig(
  input: File = null,
  modelName: String = "",
  nEpochs: Int = 1,
  hiddenSize: Int = 128,
  count: Int = 100,
  resume: Boolean = false
)

object TrainConfig {
  val parser = new OptionParser[TrainConfig]("Train") {
    head("lstmrecommender Train", "1.0")

    opt[File]('i', "input")
      .required()
      .valueName("<dir>")
      .action( (x, c) => c.copy(input = x) )
      .text("The directory with training data.")

    opt[Int]('e', "epoch")
      .action( (x, c) => c.copy(nEpochs = x) )
      .text("Number of times to go over whole training set.")

    opt[Int]('h', "hidden")
      .action( (x, c) => c.copy(hiddenSize = x) )
      .text("The size of the hidden layers.")

    opt[String]('o', "output")
      .required()
      .valueName("<modelName>")
      .action( (x, c) => c.copy(modelName = x) )
      .text("Name of trained model file.")

    opt[Int]('c', "count")
      .valueName("<count>")
      .action( (x, c) => c.copy(count = x) )
      .text("Number of examples to train on. 0.8 split")

    opt[Boolean]('r', "resume")
      .action( (x, c) => c.copy(resume = x) )
      .text("Resume learning from previous saved model.")
  }

  def parse(args: Array[String]): Option[TrainConfig] = {
    parser.parse(args, TrainConfig())
  }
}

object Train {
  private val log = LoggerFactory.getLogger(getClass)

  private def embedding(in: Int, out: Int): EmbeddingLayer =
    new EmbeddingLayer.Builder()
      .nIn(in)
      .nOut(out)
      .build()

  private def dense(in: Int, out: Int, dropOut: Boolean = true): DenseLayer =
    new DenseLayer.Builder()
      .nIn(in)
      .nOut(out)
      .activation(Activation.RELU)
      .build()

  private def lstm(nIn: Int, size: Int): GravesLSTM =
    new GravesLSTM.Builder()
      .nIn(nIn)
      .nOut(size)
      .activation(Activation.SOFTSIGN)
      .build()

  private def batchNorm(in: Int, out: Int): BatchNormalization =
    new BatchNormalization.Builder()
      .nIn(in)
      .nOut(out)
      .activation(Activation.RELU)
      .build()

  private def output(nIn: Int, nOut: Int): RnnOutputLayer =
    new RnnOutputLayer.Builder()
      .nIn(nIn)
      .nOut(nOut)
      .activation(Activation.SOFTMAX)
      .lossFunction(LossFunctions.LossFunction.MCXENT)
      .build()

  private def net(itemTypeCount: Int, countryTypeCount: Int, hiddenSize: Int) = new NeuralNetConfiguration.Builder()
    .seed(42)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .iterations(1)
    .learningRate(0.001)
    .weightInit(WeightInit.XAVIER)
    .updater(Updater.NESTEROVS)
    .momentum(0.9)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    .gradientNormalizationThreshold(1.0)
    .graphBuilder()
    .addInputs("itemIn", "countryIn", "monthIn", "weekdayIn")
    .setInputTypes(
      InputType.recurrent(itemTypeCount),
      InputType.recurrent(countryTypeCount),
      InputType.recurrent(12),
      InputType.recurrent(8))
    .addLayer("embed", embedding(itemTypeCount, hiddenSize), "itemIn")
    .addLayer("country", embedding(countryTypeCount, 10), "countryIn")
    .addLayer("month", dense(12, 12), "monthIn")
    .addLayer("weekday", dense(8, 8), "weekdayIn")
    .addLayer("bnEmbed", batchNorm(hiddenSize, hiddenSize), "embed")
    .addLayer("bnCountry", batchNorm(10, 10), "country")
    .addLayer("bnMonth", batchNorm(12, 12), "month")
    .addLayer("bnWeekday", batchNorm(8, 8), "weekday")
    .addLayer("lstm1", lstm(hiddenSize + 10 + 12 + 8, hiddenSize),  "bnEmbed", "bnCountry", "bnMonth", "bnWeekday")
    .addLayer("lstm2", lstm(hiddenSize, hiddenSize), "lstm1")
    .addLayer("dense", dense(hiddenSize, hiddenSize), "lstm2")
    .addLayer("bn", batchNorm(hiddenSize, hiddenSize), "dense")
    .addLayer("labelOut", output(hiddenSize, itemTypeCount), "bn")
    .setOutputs("labelOut")
    .build()

  def main(args: Array[String]): Unit = {
    TrainConfig.parse(args) match {
      case Some(config) =>
        log.info("Starting training")

        train(config)

        log.info("Training finished.")
      case _ =>
        log.error("Invalid arguments.")
    }
  }

  private def train(c: TrainConfig): Unit = {
    val sessionsDir = new File(c.input.getAbsolutePath + "/sessions/")
    val itemMapFile = new File(c.input.getAbsolutePath + "/items.csv")
    val countryMapFile = new File(c.input.getAbsolutePath + "/countries.csv")

    val (numClasses, numCountries, trainData) = DataIterators.onlineRetailCsv(
      sessionsDir,
      itemMapFile,
      countryMapFile,
      1,
      (c.count * 0.8).toInt
    )
    val (_, _, testData) = DataIterators.onlineRetailCsv(
      sessionsDir,
      itemMapFile,
      countryMapFile,
      (c.count * 0.8 + 1).toInt,
      c.count
    )

    log.info("Data Loaded")

    val model = if (c.resume) {
      ModelSerializer.restoreComputationGraph(c.modelName, true)
    } else {
      val conf = net(numClasses, numCountries, c.hiddenSize)
      val graph = new ComputationGraph(conf)
      graph.init()
      graph
    }

    val uiServer = UIServer.getInstance()
    val statsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    model.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage, 1))

    for (i <- 0 until c.nEpochs) {
      log.info(s"Starting epoch $i of ${c.nEpochs}")
      /*while (trainData.hasNext) {
        val ds = trainData.next()

        log.debug(s"Features: ${ds.getFeatures.mkString("\n")}")
        log.debug(s"Labels: ${ds.getLabels.mkString("\n")}")

        model.fit(ds)
      }*/
      model.fit(trainData)

      log.info(s"Finished epoch $i")
      trainData.reset()
    }

    val eval = evaluate(model, testData)
    log.info(eval.stats(true))

    ModelSerializer.writeModel(model, c.modelName, true)
    Files.write(Paths.get(c.modelName + ".eval"), eval.stats().getBytes)

    log.info(s"Model saved to: ${c.modelName}")
  }

  def evaluate(model: ComputationGraph, iterator: MultiDataSetIterator, topN: Int = 20): Evaluation = {
    val e = new Evaluation(null, topN)

    while (iterator.hasNext) {
      val next = iterator.next()
      val features = next.getFeatures
      val labels = next.getLabels
      var out = model.output(false, features :_*)
      e.evalTimeSeries(labels(0), out(0))
    }

    e
  }
}
