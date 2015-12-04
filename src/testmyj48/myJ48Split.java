package testmyj48;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

import java.util.Enumeration;

public class myJ48Split {
  /** Distribution of class values. */
  protected Distribution m_distribution;
  /** Number of created subsets. */
  protected int m_numSubsets;
  /** Desired number of branches. */
  private int m_complexityIndex;
  /** Attribute to split on. */
  private int m_attIndex;
  /** Minimum number of objects in a split.   */
  private int m_minNoObj;
  /** Value of split point. */
  private double m_splitPoint;
  /** InfoGain of split. */
  private double m_infoGain;
  /** GainRatio of split.  */
  private double m_gainRatio;
  /** The sum of the weights of the instances. */
  private double m_sumOfWeights;
  /** Number of split points. */
  private int m_index;
//  /** Static reference to splitting criterion. */
//  private static InfoGainSplitCrit infoGainCrit = new InfoGainSplitCrit();
//  /** Static reference to splitting criterion. */
//  private static GainRatioSplitCrit gainRatioCrit = new GainRatioSplitCrit();
//
  public myJ48Split(int attIndex,int minNoObj, double sumOfWeights) {
    // Get index of attribute to split on.
    m_attIndex = attIndex;
    // Set minimum number of objects.
    m_minNoObj = minNoObj;
    // Set the sum of the weights
    m_sumOfWeights = sumOfWeights;
  }

  public final int numSubsets() {
	  return m_numSubsets;
  }

  public void buildClassifier(Instances trainInstances) throws Exception {
    // Initialize the remaining instance variables.
    m_numSubsets = 0;
    m_splitPoint = Double.MAX_VALUE;
    m_infoGain = 0;
    m_gainRatio = 0;
    // Different treatment for enumerated and numeric
    // attributes.
    if (trainInstances.attribute(m_attIndex).isNominal()) {
      m_complexityIndex = trainInstances.attribute(m_attIndex).numValues();
      m_index = m_complexityIndex;
      handleEnumeratedAttribute(trainInstances);
    }else{
      m_complexityIndex = 2;
      m_index = 0;
      trainInstances.sort(trainInstances.attribute(m_attIndex));
      handleNumericAttribute(trainInstances);
    }
  }

  public final int attIndex() {
    return m_attIndex;
  }

  public final double classProb(int classIndex,Instance instance,int theSubset) throws Exception {
    if (theSubset <= -1) {
      double [] weights = weights(instance);
      if (weights == null) {
    	  return m_distribution.prob(classIndex);
      } else {
		double prob = 0;
		for (int i = 0; i < weights.length; i++) {
		  prob += weights[i] * m_distribution.prob(classIndex, i);
		}
		return prob;
      }
    } else {
      if (Utils.gr(m_distribution.perBag(theSubset), 0)) {
    	  return m_distribution.prob(classIndex, theSubset);
      } else {
    	  return m_distribution.prob(classIndex);
      }
    }
  }
// 
//  /**
//   * Returns coding cost for split (used in rule learner).
//   */
//  public final double codingCost() {
//    return Utils.log2(m_index);
//  }
// 
  public final Distribution distribution() {
    return m_distribution;
  }

  public final double gainRatio() {
    return m_gainRatio;
  }

  private void handleEnumeratedAttribute(Instances trainInstances) throws Exception {
    
    Instance instance;

    m_distribution = new Distribution(m_complexityIndex, trainInstances.numClasses());
    
    // Only Instances with known values are relevant.
    Enumeration enu = trainInstances.enumerateInstances();
    while (enu.hasMoreElements()) {
      instance = (Instance) enu.nextElement();
      if (!instance.isMissing(m_attIndex))
	m_distribution.add((int)instance.value(m_attIndex),instance);
    }
    
    // Check if minimum number of Instances in at least two
    // subsets.
    if (m_distribution.check(m_minNoObj)) {
      m_numSubsets = m_complexityIndex;
      m_infoGain = infoGainsplitCritValue(m_distribution,m_sumOfWeights);
      m_gainRatio = gainRatiosplitCritValue(m_distribution,m_sumOfWeights,m_infoGain);
    }
  }

  private void handleNumericAttribute(Instances trainInstances) throws Exception {
    int firstMiss;
    int next = 1;
    int last = 0;
    int splitIndex = -1;
    double currentInfoGain;
    double defaultEnt;
    double minSplit;
    Instance instance;
    int i;
    // Current attribute is a numeric attribute.
    m_distribution = new Distribution(2,trainInstances.numClasses());
    // Only Instances with known values are relevant.
    Enumeration enu = trainInstances.enumerateInstances();
    i = 0;
    while (enu.hasMoreElements()) {
      instance = (Instance) enu.nextElement();
      if (instance.isMissing(m_attIndex))
    	  break;
      m_distribution.add(1,instance);
      i++;
    }
    firstMiss = i;
    // Compute minimum number of Instances required in each
    // subset.
    minSplit =  0.1*(m_distribution.total())/((double)trainInstances.numClasses());
    if (Utils.smOrEq(minSplit,m_minNoObj))
      minSplit = m_minNoObj;
    else
      if (Utils.gr(minSplit,25)) 
    	  minSplit = 25;
    // Enough Instances with known values?
    if (Utils.sm((double)firstMiss,2*minSplit))
      return;
    // Compute values of criteria for all possible split
    // indices.
    defaultEnt = oldEnt(m_distribution);
    while (next < firstMiss) {
      if (trainInstances.instance(next-1).value(m_attIndex)+1e-5 < 
    	  trainInstances.instance(next).value(m_attIndex))
      {
		// Move class values for all Instances up to next 
		// possible split point.
		m_distribution.shiftRange(1,0,trainInstances,last,next);
		// Check if enough Instances in each subset and compute
		// values for criteria.
		if (Utils.grOrEq(m_distribution.perBag(0),minSplit) &&
		    Utils.grOrEq(m_distribution.perBag(1),minSplit))
		{
		  currentInfoGain = infoGainsplitCritValue(m_distribution,m_sumOfWeights,defaultEnt);
		  if (Utils.gr(currentInfoGain,m_infoGain)) {
		    m_infoGain = currentInfoGain;
		    splitIndex = next-1;
		  }
		  m_index++;
		}
		last = next;
      }
      next++;
    }
    // Was there any useful split?
    if (m_index == 0)
      return;
    // Compute modified information gain for best split.
    m_infoGain = m_infoGain-(Utils.log2(m_index)/m_sumOfWeights);
    if (Utils.smOrEq(m_infoGain,0))
      return;
    // Set instance variables' values to values for
    // best split.
    m_numSubsets = 2;
    m_splitPoint = (trainInstances.instance(splitIndex+1).value(m_attIndex)+
    				trainInstances.instance(splitIndex).value(m_attIndex))/2;
    // In case we have a numerical precision problem we need to choose the
    // smaller value
    if (m_splitPoint == trainInstances.instance(splitIndex + 1).value(m_attIndex)) {
      m_splitPoint = trainInstances.instance(splitIndex).value(m_attIndex);
    }
    // Restore distributioN for best split.
    m_distribution = new Distribution(2,trainInstances.numClasses());
    m_distribution.addRange(0,trainInstances,0,splitIndex+1);
    m_distribution.addRange(1,trainInstances,splitIndex+1,firstMiss);
    // Compute modified gain ratio for best split.
    m_gainRatio = gainRatiosplitCritValue(m_distribution,m_sumOfWeights,m_infoGain);
  }

  public final double infoGain() {
    return m_infoGain;
  }
//
//  /**
//   * Prints left side of condition..
//   *
//   * @param data training set.
//   */
//  public final String leftSide(Instances data) {
//
//    return data.attribute(m_attIndex).name();
//  }
//
//  /**
//   * Prints the condition satisfied by instances in a subset.
//   *
//   * @param index of subset 
//   * @param data training set.
//   */
//  public final String rightSide(int index,Instances data) {
//
//    StringBuffer text;
//
//    text = new StringBuffer();
//    if (data.attribute(m_attIndex).isNominal())
//      text.append(" = "+
//		  data.attribute(m_attIndex).value(index));
//    else
//      if (index == 0)
//	text.append(" <= "+
//		    Utils.doubleToString(m_splitPoint,6));
//      else
//	text.append(" > "+
//		    Utils.doubleToString(m_splitPoint,6));
//    return text.toString();
//  }
//  
//  /**
//   * Returns a string containing java source code equivalent to the test
//   * made at this node. The instance being tested is called "i".
//   *
//   * @param index index of the nominal value tested
//   * @param data the data containing instance structure info
//   * @return a value of type 'String'
//   */
//  public final String sourceExpression(int index, Instances data) {
//
//    StringBuffer expr = null;
//    if (index < 0) {
//      return "i[" + m_attIndex + "] == null";
//    }
//    if (data.attribute(m_attIndex).isNominal()) {
//      expr = new StringBuffer("i[");
//      expr.append(m_attIndex).append("]");
//      expr.append(".equals(\"").append(data.attribute(m_attIndex)
//				     .value(index)).append("\")");
//    } else {
//      expr = new StringBuffer("((Double) i[");
//      expr.append(m_attIndex).append("])");
//      if (index == 0) {
//	expr.append(".doubleValue() <= ").append(m_splitPoint);
//      } else {
//	expr.append(".doubleValue() > ").append(m_splitPoint);
//      }
//    }
//    return expr.toString();
//  }  
//
  public final void setSplitPoint(Instances allInstances) {
    double newSplitPoint = -Double.MAX_VALUE;
    double tempValue;
    Instance instance;
    if ((allInstances.attribute(m_attIndex).isNumeric()) && (m_numSubsets > 1)) {
      Enumeration enu = allInstances.enumerateInstances();
      while (enu.hasMoreElements()) {
		instance = (Instance) enu.nextElement();
		if (!instance.isMissing(m_attIndex)) {
		  tempValue = instance.value(m_attIndex);
		  if (Utils.gr(tempValue,newSplitPoint) && Utils.smOrEq(tempValue,m_splitPoint))
		    newSplitPoint = tempValue;
		}
      }
      m_splitPoint = newSplitPoint;
    }
  }
//  
//  /**
//   * Returns the minsAndMaxs of the index.th subset.
//   */
//  public final double [][] minsAndMaxs(Instances data, double [][] minsAndMaxs,
//				       int index) {
//
//    double [][] newMinsAndMaxs = new double[data.numAttributes()][2];
//    
//    for (int i = 0; i < data.numAttributes(); i++) {
//      newMinsAndMaxs[i][0] = minsAndMaxs[i][0];
//      newMinsAndMaxs[i][1] = minsAndMaxs[i][1];
//      if (i == m_attIndex)
//	if (data.attribute(m_attIndex).isNominal())
//	  newMinsAndMaxs[m_attIndex][1] = 1;
//	else
//	  newMinsAndMaxs[m_attIndex][1-index] = m_splitPoint;
//    }
//
//    return newMinsAndMaxs;
//  }
//  
  public void resetDistribution(Instances data) throws Exception {
    Instances insts = new Instances(data, data.numInstances());
    for (int i = 0; i < data.numInstances(); i++) {
      if (whichSubset(data.instance(i)) > -1) {
    	  insts.add(data.instance(i));
      }
    }
    Distribution newD = new Distribution(insts, this);
    newD.addInstWithUnknown(data, m_attIndex);
    m_distribution = newD;
  }

  public double [] weights(Instance instance) {
    double [] weights;
    int i;
    if (instance.isMissing(m_attIndex)) {
      weights = new double [m_numSubsets];
      for (i=0;i<m_numSubsets;i++)
    	  weights [i] = m_distribution.perBag(i)/m_distribution.total();
      return weights;
    }else{
      return null;
    }
  }

  public int whichSubset(Instance instance) throws Exception {
    if (instance.isMissing(m_attIndex))
      return -1;
    else{
      if (instance.attribute(m_attIndex).isNominal())
    	  return (int)instance.value(m_attIndex);
      else
    	  if (Utils.smOrEq(instance.value(m_attIndex),m_splitPoint))
    		  return 0;
    	  else
    		  return 1;
    }
  }

  public final Instances [] split(Instances data) throws Exception { 
    Instances [] instances = new Instances [m_numSubsets];
    double [] weights;
    double newWeight;
    Instance instance;
    int subset, i, j;
    for (j=0;j<m_numSubsets;j++)
      instances[j] = new Instances((Instances)data, data.numInstances());
    for (i = 0; i < data.numInstances(); i++) {
      instance = ((Instances) data).instance(i);
      weights = weights(instance);
      subset = whichSubset(instance);
      if (subset > -1)
    	  instances[subset].add(instance);
      else
		for (j = 0; j < m_numSubsets; j++)
		  if (Utils.gr(weights[j],0)) {
		    newWeight = weights[j]*instance.weight();
		    instances[j].add(instance);
		    instances[j].lastInstance().setWeight(newWeight);
		  }
    }
    for (j = 0; j < m_numSubsets; j++)
      instances[j].compactify();
    return instances;
  }

  public final boolean checkModel() {
    if (m_numSubsets > 0)
      return true;
    else
      return false;
  }

  public final double logFunc(double num) {
	// Constant hard coded for efficiency reasons
    if (num < 1e-6)
      return 0;
    else
      return num*Math.log(num)/Math.log(2);
  }

  private final double oldEnt(Distribution bags) {
    double returnValue = 0;
    int j;
    for (j=0;j<bags.numClasses();j++)
      returnValue = returnValue+logFunc(bags.perClass(j));
    return logFunc(bags.total())-returnValue;
  }

  private final double newEnt(Distribution bags) {
    double returnValue = 0;
    int i,j;
    for (i=0;i<bags.numBags();i++){
      for (j=0;j<bags.numClasses();j++)
    	  returnValue = returnValue+logFunc(bags.perClassPerBag(i,j));
      returnValue = returnValue-logFunc(bags.perBag(i));
    }
    return -returnValue;
  }

  private final double splitEnt(Distribution bags,double totalnoInst){
    double returnValue = 0;
    double noUnknown;
    int i;
    noUnknown = totalnoInst-bags.total();
    if (Utils.gr(bags.total(),0)){
      for (i=0;i<bags.numBags();i++)
	returnValue = returnValue-logFunc(bags.perBag(i));
      returnValue = returnValue-logFunc(noUnknown);
      returnValue = returnValue+logFunc(totalnoInst);
    }
    return returnValue;
  }

  private final double gainRatiosplitCritValue(Distribution bags, double totalnoInst, double numerator){
	double denumerator;
	// Compute split info.
	denumerator = splitEnt(bags,totalnoInst);	
	// Test if split is trivial.
	if (Utils.eq(denumerator,0))
		return 0;
	denumerator = denumerator/totalnoInst;
	return numerator/denumerator;
  }

  public final double infoGainsplitCritValue(Distribution bags,double totalNoInst, double oldEnt) {
	double numerator;
	double noUnknown;
	double unknownRate;
	noUnknown = totalNoInst-bags.total();
	unknownRate = noUnknown/totalNoInst;
	numerator = (oldEnt-newEnt(bags));
	numerator = (1-unknownRate)*numerator;	
	// Splits with no gain are useless.
	if (Utils.eq(numerator,0))
		return 0;
	return numerator/bags.total();
  }
  public final double infoGainsplitCritValue(Distribution bags, double totalNoInst) {
    double numerator;
    double noUnknown;
    double unknownRate;
    noUnknown = totalNoInst-bags.total();
    unknownRate = noUnknown/totalNoInst;
    numerator = (oldEnt(bags)-newEnt(bags));
    numerator = (1-unknownRate)*numerator;
    // Splits with no gain are useless.
    if (Utils.eq(numerator,0))
      return 0;
    return numerator/bags.total();
  }
}
