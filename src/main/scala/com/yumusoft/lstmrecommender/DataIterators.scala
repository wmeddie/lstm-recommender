package com.yumusoft.lstmrecommender

import java.io.File
import java.util

import org.datavec.api.records.reader.impl.csv.{CSVRecordReader, CSVSequenceRecordReader}
import org.datavec.api.split.{FileSplit, NumberedFileInputSplit}
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.Writable
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode
import org.deeplearning4j.datasets.datavec.{RecordReaderDataSetIterator, RecordReaderMultiDataSetIterator, SequenceRecordReaderDataSetIterator}
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, NormalizerStandardize}

object DataIterators {
  def onlineRetailCsv(dir: File, itemDic: File, countryDic: File, start: Int, end: Int)
      : (Int, Int, RecordReaderMultiDataSetIterator) = {

    val itemReader = new CSVSequenceRecordReader()
    itemReader.initialize(new NumberedFileInputSplit(dir.getAbsolutePath + "/Input_Item%d.csv", start, end))

    val countryReader = new CSVSequenceRecordReader()
    countryReader.initialize(new NumberedFileInputSplit(dir.getAbsolutePath + "/Input_Country%d.csv", start, end))

    val labelReader = new CSVSequenceRecordReader()
    labelReader.initialize(new NumberedFileInputSplit(dir.getAbsolutePath + "/Label_%d.csv", start, end))

    val itemDicReader = new CSVRecordReader(0, ",")
    itemDicReader.initialize(new FileSplit(itemDic))

    var last = itemDicReader.next()
    while (itemDicReader.hasNext) {
      last = itemDicReader.next()
    }

    val numClasses = last.get(1).toInt

    val countryDicReader = new CSVRecordReader(0, ",")
    countryDicReader.initialize(new FileSplit(countryDic))

    last = countryDicReader.next()
    while (countryDicReader.hasNext) {
      last = countryDicReader.next()
    }
    val numCountries = last.get(1).toInt

    val iter = new RecordReaderMultiDataSetIterator.Builder(16)
      .addSequenceReader("itemIn", itemReader)
      .addSequenceReader("countryIn", countryReader)
      .addSequenceReader("labelOut", labelReader)
      .addInput("itemIn")
      .addInput("countryIn")
      .addOutputOneHot("labelOut", 0, numClasses)
      .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END)
      .build()

    /*val iter = new SequenceRecordReaderDataSetIterator(
      rr1,
      rr2,
      32,
      numClasses,
      false,
      SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
    )*/

    (numClasses, numCountries, iter)
  }
}
