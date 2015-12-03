package wekamiddle;

import weka.core.Instance;
import weka.core.Instances;

public final class myJ48NoSplit extends myJ48Split{
	/** for serialization */
	private static final long serialVersionUID = 8020611061602844662L;

	public myJ48NoSplit(myJ48Distribution distribution){
		super(-1,-1,-1);
		m_distribution = new myJ48Distribution(distribution);
		m_numSubsets = 1;
	}

	public final void buildClassifier(Instances instances) throws Exception {
		m_distribution = new myJ48Distribution(instances);
		m_numSubsets = 1;
	}

	public int whichSubset(Instance instance){
		return 0;
	}

	public double [] weights(Instance instance){
		return null;
	}
}
