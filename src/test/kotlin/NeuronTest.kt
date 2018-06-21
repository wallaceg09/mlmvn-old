import com.wallace.intelligence.mlmvn.Neuron
import org.apache.commons.math3.complex.Complex
import java.lang.AssertionError
import kotlin.test.Test
import kotlin.test.assertTrue

class NeuronTest {
    val inputs = listOf(
            Complex(0.1, 0.1),
            Complex(0.2, 0.2),
            Complex(0.3, 0.3))

    val bias = Complex(0.4, 0.4)

    val weights = listOf(
            Complex(0.5, 0.5),
            Complex(0.6, 0.6),
            Complex(0.7, 0.7)
    )

    @Test
    fun `no exception should be raised when Neuron inputs are valid`() {
        val neuron = Neuron(inputs, bias, weights)
    }

    @Test(expected = AssertionError::class)
    fun `exception should be raised when inputs size and weights size are not equal`() {
        val badWeights = listOf(Complex(0.8, 0.8))

        val neuron = Neuron(inputs, bias, badWeights)
    }

    @Test
    fun `weighted sum should be correct`() {
        val neuron = Neuron(inputs, bias, weights)

        assertTrue { Complex.equals(neuron.weightedSum, Complex(0.4, 1.16), 1e-6) }
    }
}