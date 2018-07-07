package com.wallace.intelligence.mlmvn.activation

import com.wallace.intelligence.mlmvn.Neuron
import com.wallace.math.complex.div
import org.apache.commons.math3.complex.Complex

/**
 * [ActivationFunction] for Continuous [Neuron]s.
 * A Continuous Neuron is one such that its outputs are expected to be any arbitrary point on the unit circle.
 */
class ContinuousActivationFunction : ActivationFunction {
    override fun activate(neuron: Neuron): Complex {
        return neuron.weightedSum / neuron.weightedSum.abs()
    }
}