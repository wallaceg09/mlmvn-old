package com.wallace.intelligence.mlmvn.activation

import com.wallace.intelligence.mlmvn.Neuron
import org.apache.commons.math3.complex.Complex

/**
 * The function that activates a given [Neuron].
 */
interface ActivationFunction {
    /**
     * Activates the specific [Neuron].
     */
    fun activate(neuron: Neuron): Complex
}