package com.yumusoft.lstmrecommender

import java.io.File
import java.util

import org.datavec.api.records.reader.impl.csv.{CSVRecordReader, CSVSequenceRecordReader}
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.{RecordReaderDataSetIterator, SequenceRecordReaderDataSetIterator}
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, NormalizerStandardize}

object DataIterators {
  def onlineRetailCsv(f: File): (SequenceRecordReaderDataSetIterator, SequenceRecordReaderDataSetIterator) = {

    val Array(train, test) = new FileSplit(f).sample(null, 0.7, 0.3)
    val rr1 = new CSVSequenceRecordReader()
    rr1.initialize(train)
    val rr2 = new CSVSequenceRecordReader()
    rr2.initialize(test)

    val trainIterator = new SequenceRecordReaderDataSetIterator(rr1, 32, 3712, 3712)
    val testIterator = new SequenceRecordReaderDataSetIterator(rr2, 32, 3712, 3712)

    (trainIterator, testIterator)
  }

  def irisCsv(f: File): (RecordReaderDataSetIterator, DataNormalization)  = {
    val recordReader = new CSVRecordReader(0, ",")
    recordReader.initialize(new FileSplit(f))

    val labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
    val numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
    val batchSize = 50;    //Iris data set: 150 examples total.

    val iterator = new RecordReaderDataSetIterator(
      recordReader,
      batchSize,
      labelIndex,
      numClasses)

    val normalizer = new NormalizerStandardize()

    while (iterator.hasNext) {
      normalizer.fit(iterator.next())
    }
    iterator.reset()

    iterator.setPreProcessor(normalizer)

    (iterator, normalizer)
  }
}
