package testmyj48;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

import java.util.Enumeration;

public class myJ48ModelSelect {
  /** Minimum number of objects in interval. */
  private int m_minNoObj;
  /** All the training data */
  private Instances m_allData;

  public myJ48ModelSelect(int minNoObj, Instances allData) {
    m_minNoObj = minNoObj;
    m_allData = allData;
  }

  public void cleanup() {
    m_allData = null;
  }

  public final myJ48Split selectModel(Instances data){
    double minResult;
    double currentResult;
    myJ48Split [] currentModel;
    myJ48Split bestModel = null;
    NoSplit noSplitModel = null;
    double averageInfoGain = 0;
    int validModels = 0;
    boolean multiVal = true;
    Distribution checkDistribution;
    Attribute attribute;
    double sumOfWeights;
    int i;
    try{
      // Check if all Instances belong to one class or if not
      // enough Instances to split.
      checkDistribution = new Distribution(data);
      noSplitModel = new NoSplit(checkDistribution);
      if (Utils.sm(checkDistribution.total(),2*m_minNoObj) ||
    	  Utils.eq(checkDistribution.total(),
    			  checkDistribution.perClass(checkDistribution.maxClass()))
    	 )
    	  return noSplitModel;
      // Check if all attributes are nominal and have a 
      // lot of values.
      if (m_allData != null) {
		Enumeration enu = data.enumerateAttributes();
		while (enu.hasMoreElements()) {
		  attribute = (Attribute) enu.nextElement();
		  if ((attribute.isNumeric()) ||
		      (Utils.sm((double)attribute.numValues(),
				(0.3*(double)m_allData.numInstances())))){
		    multiVal = false;
		    break;
		  }
		}
      }
      currentModel = new myJ48Split[data.numAttributes()];
      sumOfWeights = data.sumOfWeights();
      // For each attribute.
      for (i = 0; i < data.numAttributes(); i++){
		// Apart from class attribute.
		if (i != (data).classIndex()){
		  // Get models for current attribute.
		  currentModel[i] = new myJ48Split(i,m_minNoObj,sumOfWeights);
		  currentModel[i].buildClassifier(data);
		  // Check if useful split for current attribute
		  // exists and check for enumerated attributes with 
		  // a lot of values.
		  if (currentModel[i].checkModel())
		    if (m_allData != null) {
		      if ((data.attribute(i).isNumeric()) ||
		    	 (multiVal || 
		    	  Utils.sm((double)data.attribute(i).numValues(),
						   (0.3*(double)m_allData.numInstances()))))
		      {
				averageInfoGain = averageInfoGain+currentModel[i].infoGain();
				validModels++;
		      }
		    } else {
		      averageInfoGain = averageInfoGain+currentModel[i].infoGain();
		      validModels++;
		    }
		}else
		  currentModel[i] = null;
      }
      // Check if any useful split was found.
      if (validModels == 0)
    	  return noSplitModel;
      averageInfoGain = averageInfoGain/(double)validModels;
      // Find "best" attribute to split on.
      minResult = 0;
      for (i=0;i<data.numAttributes();i++){
		if ((i != (data).classIndex()) &&
		    (currentModel[i].checkModel()))
		  // Use 1E-3 here to get a closer approximation to the original
		  // implementation.
		  if ((currentModel[i].infoGain() >= (averageInfoGain-1E-3)) &&
		      Utils.gr(currentModel[i].gainRatio(),minResult)){ 
		    bestModel = currentModel[i];
		    minResult = currentModel[i].gainRatio();
		  }
      }

      // Check if useful split was found.
      if (Utils.eq(minResult,0))
    	  return noSplitModel;
      
      // Add all Instances with unknown values for the corresponding
      // attribute to the distribution for the model, so that
      // the complete distribution is stored with the model. 
      bestModel.distribution().
	  addInstWithUnknown(data,bestModel.attIndex());
      
      // Set the split point analogue to C45 if attribute numeric.
      if (m_allData != null)
    	  bestModel.setSplitPoint(m_allData);
      return bestModel;
    } catch(Exception e){
      e.printStackTrace();
    }
    return null;
  }
}
