package com.wallace.intelligence.mlmvn

import com.wallace.math.complex.plus
import com.wallace.math.complex.times
import org.apache.commons.math3.complex.Complex

data class Neuron(val inputs: List<Complex>, val bias: Complex, val weights: List<Complex>) {
    init {
        assert(inputs.size == weights.size) {"Input list and weight list must be the same size. Inputs: $inputs. Weights: $weights."}
    }

    val weightedSum: Complex by lazy {
        var sum = bias

        for((index, input) in inputs.withIndex()) {
            sum += input * weights[index]
        }

        return@lazy sum
    }
}