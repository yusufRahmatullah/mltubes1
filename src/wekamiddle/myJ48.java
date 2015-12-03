package wekamiddle;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Statistics;
import weka.core.Capabilities.Capability;

public class myJ48 extends Classifier {
	/** for serialization */
	private static final long serialVersionUID = -9118703029167010271L;
	/** The decision tree */
	private myJ48ClassifierTree m_root;
	/** Confidence level */
	private float m_CF = 0.25f;

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		myJ48ModelSelect modSelection;
		modSelection = new myJ48ModelSelect(2, instances);
		m_root = new myJ48ClassifierTree(modSelection, m_CF);
		// can classifier tree handle the data?
		getCapabilities().testWithFail(instances);
		m_root.buildClassifier(instances);
		((myJ48ModelSelect)modSelection).cleanup();
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		return m_root.classifyInstance(instance);
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public String toString() {
		if (m_root == null) {
			return "No classifier built";
		}
		return "myJ48 pruned tree\n------------------\n" + m_root.toString();
	}

	public static double addErrs(double N, double e, float CF){
		// Ignore stupid values for CF
		if (CF > 0.5) {
			System.err.println("WARNING: confidence value for pruning " +
								" too high. Error estimate not modified.");
			return 0;
		}
		// Check for extreme cases at the low end because the
		// normal approximation won't work
		if (e < 1) {
			// Base case (i.e. e == 0) from documenta Geigy Scientific
			// Tables, 6th edition, page 185
			double base = N * (1 - Math.pow(CF, 1 / N)); 
			if (e == 0) {
				return base; 
			}
			// Use linear interpolation between 0 and 1 like C4.5 does
			return base + e * (addErrs(N, 1, CF) - base);
		}
		// Use linear interpolation at the high end (i.e. between N - 0.5
		// and N) because of the continuity correction
		if (e + 0.5 >= N) {
			// Make sure that we never return anything smaller than zero
			return Math.max(N - e, 0);
		}
		// Get z-score corresponding to CF
		double z = Statistics.normalInverse(1 - CF);
		// Compute upper limit of confidence interval
		double f = (e + 0.5) / N;
		double r = (f + (z * z) / (2 * N) +
					z * Math.sqrt((f / N) - 
									(f * f / N) + 
									(z * z / (4 * N * N)))) /
					(1 + (z * z) / N);
		return (r * N) - e;
	}
}
