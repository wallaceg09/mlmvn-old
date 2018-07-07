package com.wallace.intelligence.mlmvn.activation

import com.wallace.intelligence.mlmvn.Neuron
import com.wallace.math.complex.getRoots
import com.wallace.math.complex.times
import org.apache.commons.math3.complex.Complex
import org.apache.commons.math3.complex.RootsOfUnity
import kotlin.math.PI

/**
 * [ActivationFunction] for Discrete [Neuron]s.
 * A Discrete Neuron is one such that its outputs are expected to be one of the sectors of the unit circle created by
 * segmenting said unit circle along the [numberOfSectors] roots of unity.
 */
class DiscreteActivationFunction(val numberOfSectors: Int) : ActivationFunction {

    val rootsOfUnity: List<Complex>
    val sectorPhases: List<Double>

    init {
        val helperRoots = RootsOfUnity()
        helperRoots.computeRoots(numberOfSectors)

        rootsOfUnity = helperRoots.getRoots()

        sectorPhases = List(numberOfSectors) { (2.0 * PI * it) / numberOfSectors }
    }

    override fun activate(neuron: Neuron): Complex {
        val sectorPhase = getSectorPhase(neuron.weightedSum.argument)

        return (Complex.I * sectorPhase).exp()
    }

    private fun getSectorPhase(argz: Double): Double {
        for(phase in sectorPhases.reversed()) {
            if (argz > phase) {
                return phase
            }
        }
        throw Exception("Could not determine sector phase from argument $argz")
    }
}