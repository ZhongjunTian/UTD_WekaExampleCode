package me.zhongjun.weka;
import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class NaiveTest {
    public static void run(String trainFile, String testFile) {
    	try{
	           /*
	            * 1.读入训练、测试样本
	            */
	           File file= new File(trainFile);
	           ArffLoader loader = new ArffLoader();
	           loader.setFile(file);
	           Instances trainIns = loader.getDataSet();
	           trainIns.setClassIndex(trainIns.numAttributes()-1);	      //在使用样本之前一定要首先设置instances的classIndex，否则在使用instances对象是会抛出异常
	          
	           file = new File(testFile);
	           loader.setFile(file);
	           Instances testIns = loader.getDataSet();
	           testIns.setClassIndex(testIns.numAttributes()-1);
	          
	          
	           /* 2
	            * creat classifier
	            */
	            Classifier m_classifier = new NaiveBayes();//用以建立一个naive bayes分类器
	            m_classifier.buildClassifier(trainIns); //训练   
	          
	           /*
	            * 3.使用测试样本测试分类器的学习效果
	            * Evaluation: Class for evaluating machine learning models
	            * 即它是用于检测分类模型的类
	            */
		        Instance testInst;
	           Evaluation testingEvaluation = new Evaluation(testIns);
	           int length = testIns.numInstances();
	           for (int i =0; i < length; i++) {
	              testInst = testIns.instance(i);
	              //通过这个方法来用每个测试样本测试分类器的效果
	              testingEvaluation.evaluateModelOnceAndRecordPrediction(
	            		  m_classifier, testInst);
	           }
	          
	           /*
	            * 4.打印分类结果
	            * 其它的一些信息我们可以通过Evaluation对象的其它方法得到
	            */
	           System.out.println( "分类器的正确率：" + (1- testingEvaluation.errorRate()));
	       }catch(Exception e){
	           e.printStackTrace();
	       }
   }
}
