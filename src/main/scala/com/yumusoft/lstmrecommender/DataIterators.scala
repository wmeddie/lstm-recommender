package com.yumusoft.lstmrecommender

import java.io.File
import java.util

import org.datavec.api.records.reader.impl.csv.{CSVRecordReader, CSVSequenceRecordReader}
import org.datavec.api.split.{FileSplit, NumberedFileInputSplit}
import org.datavec.api.transform.schema.Schema
import org.deeplearning4j.datasets.datavec.{RecordReaderDataSetIterator, SequenceRecordReaderDataSetIterator}
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, NormalizerStandardize}

object DataIterators {
  def onlineRetailCsv(f: File): SequenceRecordReaderDataSetIterator = {

    val rr1 = new CSVSequenceRecordReader()
    rr1.initialize(new NumberedFileInputSplit(f.getAbsolutePath + "/Input_%d.csv", 1, 100))
    val rr2 = new CSVSequenceRecordReader()
    rr2.initialize(new NumberedFileInputSplit(f.getAbsolutePath + "/Label_%d.csv", 1, 100))
    //val featuresIterator = new SequenceRecordReaderDataSetIterator(rr1, 1, 3713, 1)
    //val labelsIterator = new SequenceRecordReaderDataSetIterator(rr2, 1, 3713, 1)

    new SequenceRecordReaderDataSetIterator(rr1, rr2, 32, 3901, false)
  }
}
