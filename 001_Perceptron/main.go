package main

import (
	"math/rand"
	"github.com/appliedgo/perceptron/draw"
	"time"
	"fmt"
)

var a, b int32

type Perceptron struct {
	bias float64
	weights []float64
}

func NewPerceptron(n int) *Perceptron {
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = rand.Float64()*2 - 1
	}
	return &Perceptron{
		bias: rand.Float64()*2 - 1,
		weights: w,
	}
}

func (p *Perceptron) heaviside(f float64) int {
	if f < 0 {
		return 0
	}
	return 1
}

func (p *Perceptron) Process(inputs []int32) int {
	sum := p.bias
	for i, input := range inputs {
		sum += float64(input) * p.weights[i]
	}
	return p.heaviside(sum)
}

func (p *Perceptron) Adjust(inputs []int32, delta int, learningRate float64) {
	for i, input := range inputs {
		p.weights[i] += float64(input) * float64(delta) * learningRate
	}
	p.bias += float64(delta) * learningRate
}

// linear function equation
func f(x int32) int32 {
	return a*x + b
}

func isAboveLine(point []int32, f func(int32) int32) int {
	x := point[0]
	y := point[1]
	if y > f(x) {
		return 1
	}
	return 0
}

func train(p *Perceptron, iters int, rate float64) {
	for i := 0; i < iters; i++ {
		point := []int32{rand.Int31n(201) - 101, rand.Int31n(201) - 101}
		actual := p.Process(point)
		expected := isAboveLine(point, f)
		delta := expected - actual
		p.Adjust(point, delta, rate)
	}
}

func verify(p *Perceptron, points int) int {
	var correctAnswer int
	c := draw.NewCanvas()
	for i := 0; i < points; i++ {
		point := []int32{rand.Int31n(201) - 101, rand.Int31n(201) - 101}
		result := p.Process(point)
		if result == isAboveLine(point, f) {
			correctAnswer++
		}
		c.DrawPoint(point[0], point[1], result == 1)
	}
	c.DrawLinearFunction(a, b)
	c.Save()
	return correctAnswer
}

func main() {
	rand.Seed(time.Now().UnixNano())
	a = rand.Int31n(11) - 6
	b = rand.Int31n(101) - 51
	p := NewPerceptron(2)
	iters := 10000
	points := 40000
	var rate float64 = 0.3

	train(p, iters, rate)
	successRate := verify(p, points)
	fmt.Printf("%.1f%% of the answers were correct.\n", float64(float64(successRate) / float64(points) * 100))
}
