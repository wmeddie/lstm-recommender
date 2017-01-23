package com.yumusoft.lstmrecommender

import java.io.File
import java.util

import org.datavec.api.records.reader.impl.csv.{CSVRecordReader, CSVSequenceRecordReader}
import org.datavec.api.split.{FileSplit, NumberedFileInputSplit}
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.Writable
import org.deeplearning4j.datasets.datavec.{RecordReaderDataSetIterator, SequenceRecordReaderDataSetIterator}
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, NormalizerStandardize}

object DataIterators {
  def onlineRetailCsv(f: File, dictionary: File, start: Int, end: Int): (Int, SequenceRecordReaderDataSetIterator) = {

    val rr1 = new CSVSequenceRecordReader()
    rr1.initialize(new NumberedFileInputSplit(f.getAbsolutePath + "/Input_%d.csv", start, end))
    val rr2 = new CSVSequenceRecordReader()
    rr2.initialize(new NumberedFileInputSplit(f.getAbsolutePath + "/Label_%d.csv", start, end))
    //val featuresIterator = new SequenceRecordReaderDataSetIterator(rr1, 1, 3713, 1)
    //val labelsIterator = new SequenceRecordReaderDataSetIterator(rr2, 1, 3713, 1)

    val dict = new CSVRecordReader(0, ",")
    dict.initialize(new FileSplit(dictionary))

    var last = dict.next()
    while (dict.hasNext) {
      last = dict.next()
    }

    val numClasses = last.get(1).toInt

    val iter = new SequenceRecordReaderDataSetIterator(
      rr1,
      rr2,
      32,
      numClasses,
      false,
      SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
    )

    (numClasses, iter)
  }
}
