package cnn
//Referenced from : https://github.com/jonysugianto/dfa_nn
import (
	"fmt"
	"math"
	"math/rand"
)

type ConvolveNeuralNetwork struct{
	nConvPoolLayers int
	nMLP int
	class int
	CPL []ConvolutionalPoolingLayer
	Connect Connect
	MLP []MultiLayerPerceptron
	LR LogisticRegression
}


type ConvolutionalPoolingLayer struct{
	imageSize [] int
	channel int
	nKernels int
	kernelSize []int
	poolSize []int
	convolvedSize []int
	pooledSize []int
	miniBatchSize int
	activationName string

	W [][][][] float64
	Bias [] float64

	Input [][][][]float64
	Convolved [][][][]float64
	Activated [][][][]float64
	Pooled [][][][]float64
	DMaxPool [][][][]float64
	DConvolve [][][][]float64
}


type Connect struct{
	miniBatchSize int
	nKernel int
	pooledSize []int
	flattenedSize int
	hiddenLayerSize int
	Flattened [][]float64
	Unflattened [][][][]float64
}
type MultiLayerPerceptron struct{

	In int
	Out int
	dropOut bool
	dropOutPossibility float64
	miniBatchSize int
	activationName string

	W [][]float64
	Bias []float64

	Input [][]float64
	PreActivate [][]float64
	Activated [][]float64
	Delta [][]float64
	DropOutMask [][]int
}


type LogisticRegression struct{

	in int
	out int
	miniBatchSize int

	W [][]float64
	Bias []float64

	Output [][]float64
	Delta [][]float64
	CrossEntropy float64
	AccuracyCount int
	PredictedLabel [][]int
}

func CPLConstruct(self *ConvolutionalPoolingLayer,
	imageSize      []int,
	channel          int,
	nKernels         int,
	kernelSize     []int,
	poolSize       []int,
	convolvedSize  []int,
	pooledSize     []int,
	miniBatchSize    int,
	activationName   string){

	self.imageSize = imageSize
	self.channel = channel
	self.nKernels = nKernels
	self.kernelSize = kernelSize
	self.poolSize = poolSize
	self.convolvedSize = convolvedSize
	self.pooledSize = pooledSize
	self.miniBatchSize = miniBatchSize
	self.activationName = activationName

	var(
		in int = channel * kernelSize[0] * kernelSize[1]
		out int = nKernels * kernelSize[0] * kernelSize[1] / (poolSize[0] * poolSize[1])
		randBoundary float64 = math.Sqrt(6.0 / float64(in + out))
	)

	self.W = make([][][][]float64, nKernels)
	for i := 0; i < nKernels; i++{
		self.W[i] = make([][][]float64, channel)

		for j := 0; j < channel; j++{
			self.W[i][j] = make([][]float64, kernelSize[0])

			for k := 0; k < kernelSize[0]; k++{
				self.W[i][j][k] = make([]float64, kernelSize[1])
			}
		}
	}

	for i := 0; i < nKernels; i++{
		for j := 0; j < channel; j++{
			for k := 0; k < kernelSize[0]; k++{
				for l := 0; l < kernelSize[1]; l++{
					self.W[i][j][k][l] = Uniform(-randBoundary, randBoundary)
				}
			}
		}
	}

	self.Bias = make([] float64,nKernels)
}

func CPLConfirm(self *ConvolutionalPoolingLayer){

	fmt.Println("ImageSize")
	fmt.Println(self.imageSize)
	fmt.Println("Channel")
	fmt.Println(self.channel)
	fmt.Println("NumberKernel")
	fmt.Println(self.nKernels)
	fmt.Println("KernelSize")
	fmt.Println(self.kernelSize)
	fmt.Println("PoolSize")
	fmt.Println(self.poolSize)
	fmt.Println("ConvolvedSize")
	fmt.Println(self.convolvedSize)
	fmt.Println("PooledSize")
	fmt.Println(self.pooledSize)
	fmt.Println("miniBatchSize")
	fmt.Println(self.miniBatchSize)
	fmt.Println("ActivationName")
	fmt.Println(self.activationName)
}

func DirConConstruct(self *Connect,
	miniBatchSize   int,
	nKernel         int,
	pooledSize []   int,
	flattenedSize   int,
	hiddenLayerSize int){

	self.miniBatchSize   = miniBatchSize
	self.nKernel         = nKernel
	self.pooledSize      = pooledSize
	self.flattenedSize   = flattenedSize
	self.hiddenLayerSize = hiddenLayerSize
}

func MLPConstruct(self *MultiLayerPerceptron,
	In                 int,
	Out                int,
	dropOut            bool,
	dropOutPossibility float64,
	miniBatchSize      int,
	activationName     string){

	self.In = In
	self.Out = Out
	self.dropOut = dropOut
	self.dropOutPossibility = dropOutPossibility
	self.miniBatchSize = miniBatchSize
	self.activationName = activationName

	var randomBoundary float64 = 1.0 / float64(In)
	self.W = make([][]float64, Out)
	for i := 0; i < Out; i++{
		self.W[i] = make([]float64, In)
	}
	for i := 0; i < Out; i++{
		for j := 0; j < In; j++{
			self.W[i][j] = Uniform(-randomBoundary, randomBoundary)
		}
	}

	self.Bias = make([]float64, Out)
}

func MLPConfirm(self *MultiLayerPerceptron){

	fmt.Println("In")
	fmt.Println(self.In)
	fmt.Println("Out")
	fmt.Println(self.Out)
	fmt.Println("Dropout")
	fmt.Println(self.dropOut)
	fmt.Println("DropOutPossibility")
	fmt.Println(self.dropOutPossibility)
	fmt.Println("miniBatchSize")
	fmt.Println(self.miniBatchSize)
	fmt.Println("ActivationName")
	fmt.Println(self.activationName)
}


func Construct(self *ConvolveNeuralNetwork,
	imageSize          []int,
	channel              int,
	nKernel            []int,
	kernelSizes      [][]int,
	poolSizes        [][]int,
	mlpSizes           []int,
	class                int,
	miniBatchSize        int,
	dropOut            []bool,
	dropOutPossibility []float64,
	activationName       string){

	self.nConvPoolLayers = len(nKernel)
	self.nMLP = len(mlpSizes)
	self.class = class

	fmt.Println("Construct the Convolve and Pooling layer.")
	self.CPL = make([]ConvolutionalPoolingLayer, self.nConvPoolLayers)

	var(
		inSize = make([][]int, self.nConvPoolLayers)
		convedSize = make([][]int, self.nConvPoolLayers)
		pooledSize = make([][]int, self.nConvPoolLayers)
	)
	for i := 0; i < self.nConvPoolLayers; i++{
		inSize[i] = make([]int, 2)
		convedSize[i] = make([]int, 2)
		pooledSize[i] = make([]int, 2)
	}

	for i := 0; i < self.nConvPoolLayers; i++{
		var eachChannel int

		if i == 0{
			inSize[i][0] = imageSize[0]
			inSize[i][1] = imageSize[1]
			eachChannel = channel
		}else{
			inSize[i][0] = pooledSize[i - 1][0]
			inSize[i][1] = pooledSize[i - 1][1]
			eachChannel = nKernel[i - 1]
		}
		convedSize[i][0] = inSize[i][0] - kernelSizes[i][0] + 1
		convedSize[i][1] = inSize[i][1] - kernelSizes[i][1] + 1
		pooledSize[i][0] = convedSize[i][0] / poolSizes[i][0]
		pooledSize[i][1] = convedSize[i][1] / poolSizes[i][1]

		fmt.Printf("Construct the %d layer.\n", i + 1)
		CPLConstruct(&(self.CPL[i]), inSize[i], eachChannel, nKernel[i], kernelSizes[i], poolSizes[i], convedSize[i], pooledSize[i], miniBatchSize, activationName)
		CPLConfirm(&(self.CPL[i]))
	}

	fmt.Println("-----------------------------------")
	fmt.Println("Construct the Connection.")
	flattenedSize := nKernel[self.nConvPoolLayers - 1] * pooledSize[self.nConvPoolLayers - 1][0] * pooledSize[self.nConvPoolLayers - 1][1]
	DirConConstruct((&self.Connect), miniBatchSize, nKernel[self.nConvPoolLayers - 1], pooledSize[self.nConvPoolLayers - 1], flattenedSize, mlpSizes[0])

	fmt.Println("-----------------------------------")
	fmt.Println("Construct the MultiLayerPerceptron.")
	self.MLP = make([]MultiLayerPerceptron, self.nMLP)

	for i := 0; i < self.nMLP; i++{
		var in int

		if i == 0{
			in = flattenedSize
		}else{
			in = mlpSizes[i - 1]
		}

		fmt.Printf("Construct the %d layer.\n", i + 1)
		MLPConstruct((&self.MLP[i]), in, mlpSizes[i], dropOut[i], dropOutPossibility[i], miniBatchSize, activationName)
		MLPConfirm((&self.MLP[i]))
	}

	fmt.Println("---------------------------------")
	fmt.Println("Construct the LogisticRegression.")
	LRConstruct((&self.LR), mlpSizes[self.nMLP - 1], class, miniBatchSize)
	LRConfirm(&self.LR)
	fmt.Println("---------------------------------")
}

func LRConstruct(self *LogisticRegression, in int, out int, miniBatchSize int){

	self.in = in
	self.out = out
	self.miniBatchSize = miniBatchSize

	var randomBoundary float64 = 1.0 / float64(in)
	self.W = make([][]float64, out)
	for i := 0; i < out; i++{
		self.W[i] = make([]float64, in)
	}
	for i := 0; i < out; i++{
		for j := 0; j < in; j++{
			self.W[i][j] = Uniform(-randomBoundary, randomBoundary)
		}
	}

	self.Bias = make([]float64, out)
}

func LRConfirm(self *LogisticRegression){

	fmt.Println("In")
	fmt.Println(self.in)
	fmt.Println("Label")
	fmt.Println(self.out)
	fmt.Println("miniBatchSize")
	fmt.Println(self.miniBatchSize)
}


func Uniform(min float64, max float64) float64{

	return rand.Float64() * (max - min) + min
}

func Flatten(self *Connect, input [][][][]float64){

	var flattened = make([][]float64, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		flattened[i] = make([]float64, self.flattenedSize)
	}

	for i := 0; i < self.miniBatchSize; i++{
		index := 0

		for j := 0; j < self.nKernel; j++{
			for k := 0; k < self.pooledSize[0]; k++{
				for l := 0; l < self.pooledSize[1]; l++{

					flattened[i][index] = input[i][j][k][l]
					index += 1
				}
			}
		}
	}
	self.Flattened = flattened
}



func MPLOutput(self *MultiLayerPerceptron, input [][]float64, TrainOrTest string){

	var preActivate = make([][]float64, self.miniBatchSize)
	var activated = make([][]float64, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		preActivate[i] = make([]float64, self.Out)
		activated[i] = make([]float64, self.Out)
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.Out; j++{
			Out := 0.0

			for k := 0; k < self.In; k++{
				Out += self.W[j][k] * input[i][k]
			}
			preActivate[i][j] = Out + self.Bias[j]
			activated[i][j] = Activation(self.activationName, preActivate[i][j])
		}
	}
	self.PreActivate = preActivate
	self.Activated = activated
}

func MLPdropOut(self *MultiLayerPerceptron, TrainOrTest string){

	var dropOutMask = make([][]int, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		dropOutMask[i] = make([]int, self.Out)
	}

	if TrainOrTest == "Train"{
		for i := 0; i < self.miniBatchSize; i++{
			for j := 0; j < self.Out; j++{
				random := rand.Float64()

				if random < self.dropOutPossibility{
					dropOutMask[i][j] = 0
					self.Activated[i][j] *= float64(dropOutMask[i][j])
				}else{
					dropOutMask[i][j] = 1
					self.Activated[i][j] *= float64(dropOutMask[i][j])
				}
			}
		}
	}else if TrainOrTest == "Test"{
		for i := 0; i < self.miniBatchSize; i++{
			for j := 0; j < self.Out; j++{
				self.Activated[i][j] *= (1 - self.dropOutPossibility)
			}
		}
	}
	self.DropOutMask = dropOutMask
}

func MLPForward(self *MultiLayerPerceptron, input [][]float64, TrainOrTest string){

	if TrainOrTest == "Train"{
		self.Input = input
	}

	MPLOutput(self, input, TrainOrTest)

	if self.dropOut == true{
		MLPdropOut(self, TrainOrTest)
	}
}

func MLPBackward(self *MultiLayerPerceptron, prevDelta [][]float64, prevW [][] float64, prevLayerOut int, learningRate float64){

	var gradW = make([][]float64, self.Out)
	for i:= 0; i < self.Out; i++{
		gradW[i] = make([]float64, self.In)
	}

	var gradBias = make([]float64, self.Out)

	var delta = make([][]float64, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		delta[i] = make([]float64, self.Out)
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.Out; j++{
			for k:= 0; k < prevLayerOut; k++{

				if self.dropOut == true{
					delta[i][j] += float64(self.DropOutMask[i][j]) * prevW[k][j] * prevDelta[i][k]
				}else if self.dropOut == false{
					delta[i][j] += prevW[k][j] * prevDelta[i][k]
				}
				delta[i][j] *= Dactivation(self.activationName, self.PreActivate[i][j])

				gradBias[j] += delta[i][j]

				for l := 0; l < self.In; l++{
					gradW[j][l] += delta[i][j] * self.Input[i][l]
				}
			}
		}
	}

	for i := 0; i < self.Out; i++{
		for j := 0; j < self.In; j++{
			self.W[i][j] -= learningRate * gradW[i][j] / float64(self.miniBatchSize)
		}
		self.Bias[i] -= learningRate * gradBias[i] / float64(self.miniBatchSize)
	}
	self.Delta = delta
}

func Train(self *ConvolveNeuralNetwork, input [][][][]float64, actualLabel [][]int, learningRate float64){

	TrainOrTest := "Train"

	fmt.Println("ConvolutionPoolingLayer")
	for i := 0; i < self.nConvPoolLayers; i++{
		if i == 0{
			fmt.Printf("%d layer\n", i + 1)
			CPLForward((&self.CPL[i]), input, TrainOrTest)
		}else{
			fmt.Printf("%d layer\n", i + 1)
			CPLForward((&self.CPL[i]), self.CPL[i - 1].Pooled, TrainOrTest)
		}
	}

	fmt.Println("Conneting")
	Flatten((&self.Connect), self.CPL[self.nConvPoolLayers - 1].Pooled)

	fmt.Println("MultiLayerPerceptron")
	for i := 0; i < self.nMLP; i++{
		if i == 0{
			fmt.Printf("%d layer\n", i + 1)
			MLPForward((&self.MLP[i]), self.Connect.Flattened, TrainOrTest)
		}else{
			fmt.Printf("%d layer\n", i + 1)
			MLPForward((&self.MLP[i]), self.MLP[i - 1].Activated, TrainOrTest)
		}
	}

	fmt.Println("LogisticRegression")
	LRTrain((&self.LR), self.MLP[self.nMLP - 1].Activated, actualLabel, learningRate)

	fmt.Println("Back MLP")
	for i := self.nMLP - 1; 0 <= i; i--{
		if i == self.nMLP - 1{
			fmt.Printf("%d layer\n", i + 1)
			MLPBackward((&self.MLP[i]), self.LR.Delta, self.LR.W, self.class, learningRate)
		}else{
			fmt.Printf("$d layer\n", i + 1)
			MLPBackward((&self.MLP[i]), self.MLP[i + 1].Delta, self.MLP[i + 1].W, self.MLP[i + 1].Out, learningRate)
		}
	}

	fmt.Println("Connecting")
	Unflatten((&self.Connect), self.MLP[0].Delta, self.MLP[0].W)

	fmt.Println("Back ConvPool layer")
	for i := self.nConvPoolLayers - 1; 0 <= i; i--{
		if i == self.nConvPoolLayers - 1{
			fmt.Printf("%d layer\n", i + 1)
			CPLBackward((&self.CPL[i]), self.Connect.Unflattened, learningRate)
		}else{
			fmt.Printf("%d layer\n", i + 1)
			CPLBackward((&self.CPL[i]), self.CPL[i + 1].DConvolve, learningRate)
		}
	}
}


func Unflatten(self *Connect, input [][]float64, W [][]float64){

	var delta [][]float64
	delta = make([][]float64, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		delta[i] = make([]float64, self.flattenedSize)
	}

	var unflattened = make([][][][]float64, self.miniBatchSize)

	for i := 0; i < self.miniBatchSize; i++{
		unflattened[i] = make([][][]float64, self.nKernel)

		for j := 0; j < self.nKernel; j++{
			unflattened[i][j] = make([][]float64, self.pooledSize[0])

			for k := 0; k < self.pooledSize[0]; k++{
				unflattened[i][j][k] = make([]float64, self.pooledSize[1])
			}
		}
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.flattenedSize; j++{
			for k := 0; k < self.hiddenLayerSize; k++{

				delta[i][j] = W[k][j] * input[i][k]
			}
		}
	}

	for i := 0; i < self.miniBatchSize; i++{
		index := 0

		for j := 0; j < self.nKernel; j++{
			for k := 0; k < self.pooledSize[0]; k++{
				for l := 0; l < self.pooledSize[1]; l++{
					unflattened[i][j][k][l] = delta[i][index]
					index += 1
				}
			}
		}
	}
	self.Unflattened = unflattened
}


func SoftMax(input []float64) []float64{

	var(
		max float64 = 0.0
		sum float64 = 0.0
	)

	size := len(input)
	var output  = make([]float64, size)

	for i := 0; i < size; i++{
		if input[i] > max{
			max = input[i]
		}
	}

	for i := 0; i < size; i++{
		output[i] = math.Exp(input[i] - max)
		sum += output[i]
	}

	for i := 0; i < size; i++{
		output[i] /= sum
	}
	return output
}

func CrossEntropy(input []float64, label []int) float64{

	var output float64

	size := len(input)
	for i := 0; i < size; i++{
		output += (-1) * math.Log(input[i]) * float64(label[i])
	}

	fmt.Println("CrossEntropy")
	fmt.Println(output)

	return output
}

func LROutput(self *LogisticRegression, input [][]float64){

	var output = make([][]float64, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		output[i] = make([]float64, self.out)
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.out; j++{
			var out float64 = 0.0

			for k := 0; k < self.in; k++{
				out += self.W[j][k] * input[i][k]
			}
			output[i][j] = out + self.Bias[j]
		}
	}
	self.Output = output
}

func LRTrain(self *LogisticRegression, input [][]float64, actualLabel [][]int, learningRate float64){

	var gradW = make([][]float64, self.out)
	for i := 0; i < self.out; i++{
		gradW[i] = make([]float64, self.in)
	}

	var gradBias = make([]float64, self.out)

	var delta = make([][]float64, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		delta[i] = make([]float64, self.out)
	}

	var crossEntropy float64 = 0.0

	LROutput(self, input)
	for i := 0; i < self.miniBatchSize; i++{
		self.Output[i] = append(SoftMax(self.Output[i]))

		fmt.Println("ActualLabel")
		fmt.Println(actualLabel[i])
		fmt.Println("SoftMax")
		fmt.Println(self.Output[i])

		crossEntropy += CrossEntropy(self.Output[i], actualLabel[i])
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.out; j++{
			delta[i][j] = self.Output[i][j] - float64(actualLabel[i][j])

			gradBias[j] += delta[i][j]

			for k := 0; k < self.in; k++{
				gradW[j][k] += delta[i][j] * input[i][k]
			}
		}
	}

	for i := 0; i < self.out; i++{
		for j := 0; j < self.in; j++{
			self.W[i][j] -= learningRate * gradW[i][j] / float64(self.miniBatchSize)
		}
		self.Bias[i] -= learningRate * gradBias[i] / float64(self.miniBatchSize)
	}
	self.CrossEntropy = crossEntropy
	self.Delta = delta
}

func Convolve(self *ConvolutionalPoolingLayer, Input [][][][]float64){

	var convolved = make([][][][]float64, self.miniBatchSize)
	var activated = make([][][][]float64, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		convolved[i] = make([][][]float64, self.nKernels)
		activated[i] = make([][][]float64, self.nKernels)

		for j := 0; j < self.nKernels; j++{
			convolved[i][j] = make([][]float64, self.convolvedSize[0])
			activated[i][j] = make([][]float64, self.convolvedSize[0])

			for k := 0; k < self.convolvedSize[0]; k++{
				convolved[i][j][k] = make([]float64, self.convolvedSize[1])
				activated[i][j][k] = make([]float64, self.convolvedSize[1])
			}
		}
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j:= 0; j < self.nKernels; j++{
			for k := 0; k < self.convolvedSize[0]; k++{
				for l:= 0; l < self.convolvedSize[1]; l++{

					for m:= 0; m < self.channel; m++{
						for n:= 0; n < self.kernelSize[0]; n++{
							for o:= 0; o < self.kernelSize[1]; o++{
								convolved[i][j][k][l] += self.W[j][m][n][o] * Input[i][m][k + n][l + o] + self.Bias[j]
							}
						}
					}
					activated[i][j][k][l] = Activation(self.activationName, convolved[i][j][k][l])
				}
			}
		}
	}
	self.Convolved = convolved
	self.Activated = activated
}

func MaxPool(self *ConvolutionalPoolingLayer, Input [][][][]float64){

	var pooled = make([][][][]float64, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		pooled[i] = make([][][]float64, self.nKernels)

		for j := 0; j < self.nKernels; j++{
			pooled[i][j] = make([][]float64, self.pooledSize[0])

			for k := 0; k < self.pooledSize[0]; k++{
				pooled[i][j][k] = make([]float64, self.pooledSize[1])
			}
		}
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.nKernels; j++{
			for k := 0; k < self.pooledSize[0]; k++{
				for l := 0; l < self.pooledSize[1]; l++{
					max := 0.0

					for m:= 0; m < self.poolSize[0]; m++{
						for n := 0; n < self.poolSize[1]; n++{

							if m == 0 && n == 0{
								max = self.Activated[i][j][k * self.poolSize[0]][l * self.poolSize[1]]
								continue
							}
							if max < self.Activated[i][j][k * self.poolSize[0] + m][l * self.poolSize[1] + n]{
								max = self.Activated[i][j][k * self.poolSize[0] + m][l * self.poolSize[1] + n]
							}
						}
					}
					pooled[i][j][k][l] = max
				}
			}
		}
	}
	self.Pooled = pooled
}

func DMaxPool(self *ConvolutionalPoolingLayer, prevDelta [][][][]float64, learningRate float64){

	var dMaxPool = make([][][][]float64, self.miniBatchSize)
	for i:= 0; i < self.miniBatchSize; i++{
		dMaxPool[i] = make([][][]float64, self.nKernels)

		for j := 0; j < self.nKernels; j++{
			dMaxPool[i][j] = make([][]float64, self.convolvedSize[0])

			for k := 0; k < self.convolvedSize[0]; k++{
				dMaxPool[i][j][k] = make([]float64, self.convolvedSize[1])
			}
		}
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.nKernels; j++{
			for k := 0; k < self.pooledSize[0]; k++{
				for l := 0; l < self.pooledSize[1]; l++{
					for m:= 0; m < self.poolSize[0]; m++{
						for n := 0; n < self.poolSize[1]; n++{

							delta := 0.0

							if self.Pooled[i][j][k][l] == self.Activated[i][j][k * self.poolSize[0] + m][l * self.poolSize[1] + n]{
								delta = prevDelta[i][j][k][l]
							}
							dMaxPool[i][j][k * self.poolSize[0] + m][l * self.poolSize[1] + n] = delta
						}
					}
				}
			}
		}
	}
	self.DMaxPool = dMaxPool
}

func DConvolve(self *ConvolutionalPoolingLayer, learningRate float64){

	var gradW = make([][][][]float64, self.nKernels)
	for i := 0; i < self.nKernels; i++{
		gradW[i] = make([][][]float64, self.channel)

		for j := 0; j < self.channel; j++{
			gradW[i][j] = make([][]float64, self.kernelSize[0])

			for k := 0; k < self.kernelSize[0]; k++{
				gradW[i][j][k] = make([]float64, self.kernelSize[1])
			}
		}
	}

	var gradBias = make([]float64, self.nKernels)

	var dConvolve = make([][][][]float64, self.miniBatchSize)

	for i := 0; i < self.miniBatchSize; i++{
		dConvolve[i] = make([][][]float64, self.channel)

		for j := 0; j < self.channel; j++{
			dConvolve[i][j] = make([][]float64, self.imageSize[0])

			for k := 0; k < self.imageSize[0]; k++{
				dConvolve[i][j][k] = make([]float64, self.imageSize[1])
			}
		}
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.nKernels; j++{
			for k := 0; k < self.convolvedSize[0]; k++{
				for l := 0; l < self.convolvedSize[1]; l++{

					d :=  self.DMaxPool[i][j][k][l] * Dactivation(self.activationName, self.Convolved[i][j][k][l])
					gradBias[j] += d

					for m := 0; m < self.channel; m++{
						for n := 0; n < self.kernelSize[0]; n++{
							for o := 0; o < self.kernelSize[1]; o++{

								gradW[j][m][n][o] += d * self.Input[i][m][k + n][l + o]
							}
						}
					}
				}
			}
		}
	}

	for i := 0; i < self.nKernels; i++{
		self.Bias[i] -= learningRate * gradBias[i] / float64(self.miniBatchSize)

		for j := 0; j < self.channel; j++{
			for k := 0; k < self.kernelSize[0]; k++{
				for l := 0; l < self.kernelSize[1]; l++{

					self.W[i][j][k][l] -= learningRate * gradW[i][j][k][l] / float64(self.miniBatchSize)
				}
			}
		}
	}

	var delta float64
	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.channel; j++{
			for k := 0; k < self.imageSize[0]; k++{
				for l := 0; l < self.imageSize[1]; l++{

					for m := 0; m < self.nKernels; m++{
						for n := 0; n < self.kernelSize[0]; n++{
							for o := 0; o < self.kernelSize[1]; o++{

								if (k - (self.kernelSize[0] - 1) - n < 0) || (l - (self.kernelSize[1] - 1) - o < 0){
									delta = 0.0
								}else{
									delta = self.DMaxPool[i][m][k - (self.kernelSize[0] - 1) - n][l - (self.kernelSize[1] - 1) - o] *
										Dactivation(self.activationName, self.Convolved[i][m][k - (self.kernelSize[0] - 1) - n][l - (self.kernelSize[1] - 1) - o]) *
										self.W[m][j][n][o]
								}
								dConvolve[i][j][k][l] += delta
							}
						}
					}
				}
			}
		}
	}
	self.DConvolve = dConvolve
}
func Activation(activationName string, input float64) float64{

	var x float64
	if activationName == "ReLU"{

		if input >= 0.0{
			x = input
		}else{
			x = 0.0
		}
	}
	if activationName == "Sigmoid"{

		input *= -1
		x = 1.0 / (1 + math.Exp(input))
	}
	if activationName == "Tanh"{

		x = math.Tanh(input)
	}
	return x
}

func Dactivation(activationName string, input float64) float64{

	var x float64
	if activationName == "ReLU"{

		if input >= 0.0{
			x = 1.0
		}else{
			x = 0.0
		}
	}
	if activationName == "Sigmoid"{

		input *= -1
		x = (1 - 1.0 / (1 + math.Exp(input))) * (1.0 / (1 + math.Exp(input)))
	}
	if activationName == "Tanh"{

		x = 1 / math.Pow(math.Cosh(input), 2)
	}
	return x
}

func CPLForward(self *ConvolutionalPoolingLayer, Input [][][][]float64, TrainOrTest string){

	if TrainOrTest == "Train"{
		self.Input = Input
	}
	Convolve(self, Input)
	MaxPool(self, Input)
}

func CPLBackward(self *ConvolutionalPoolingLayer, prevDelta [][][][]float64, learningRate float64){

	DMaxPool(self, prevDelta, learningRate)
	DConvolve(self, learningRate)
}

func Test(self *ConvolveNeuralNetwork, input [][][][]float64, actualLabel [][]int){

	TrainOrTest := "Test"

	fmt.Println("ConvolutionPoolingLayer")
	for i := 0; i < self.nConvPoolLayers; i++{
		if i == 0{
			fmt.Printf("%d layer\n", i + 1)
			CPLForward((&self.CPL[i]), input, TrainOrTest)
		}else{
			fmt.Printf("%d layer\n", i + 1)
			CPLForward((&self.CPL[i]), self.CPL[i - 1].Pooled, TrainOrTest)
		}
	}

	fmt.Println("Conneting")
	Flatten((&self.Connect), self.CPL[self.nConvPoolLayers - 1].Pooled)

	fmt.Println("MultiLayerPerceptron")
	for i := 0; i < self.nMLP; i++{
		if i == 0{
			fmt.Printf("%d layer\n", i + 1)
			MLPForward((&self.MLP[i]), self.Connect.Flattened, TrainOrTest)
		}else{
			fmt.Printf("%d layer\n", i + 1)
			MLPForward((&self.MLP[i]), self.MLP[i - 1].Activated, TrainOrTest)
		}
	}

	fmt.Println("LogisticRegression")
	LRPredict((&self.LR), self.MLP[self.nMLP - 1].Activated, actualLabel)
}

func LRPredict(self *LogisticRegression, input [][]float64, actualLabel [][]int){

	var argMax []int
	argMax = make([]int, self.miniBatchSize)

	var predictedLabel = make([][]int, self.miniBatchSize)
	for i := 0; i < self.miniBatchSize; i++{
		predictedLabel[i] = make([]int, self.out)
	}

	var accuracyCount int = 0

	LROutput(self, input)
	for i := 0; i < self.miniBatchSize; i++{
		self.Output[i] = append(SoftMax(self.Output[i]))
	}

	for i := 0; i < self.miniBatchSize; i++{
		var max float64 = 0.0
		for j := 0; j < self.out; j++{
			if self.Output[i][j] > max{
				max = self.Output[i][j]
				argMax[i] = j
			}
		}
	}

	for i := 0; i < self.miniBatchSize; i++{
		for j := 0; j < self.out; j++{
			if argMax[i] == j{
				predictedLabel[i][j] = 1
			}else{
				predictedLabel[i][j] = 0
			}
		}
	}

	for i := 0; i < self.miniBatchSize; i++{

		fmt.Println()
		fmt.Println("ActualLabel")
		fmt.Println(actualLabel[i])
		fmt.Println("SoftMax")
		fmt.Println(self.Output[i])
		fmt.Println("PredictedLabel")
		fmt.Println(predictedLabel[i])

		for j := 0; j < self.out; j++{
			if (predictedLabel[i][j] == 1 && actualLabel[i][j] == 1){
				accuracyCount += 1
				fmt.Println("Predicted")
			}
		}
	}
	self.AccuracyCount  = accuracyCount
	self.PredictedLabel = predictedLabel
}
