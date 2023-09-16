<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://wallpapers.com/images/hd/4k-minimalist-mountains-9oitratl6gd996za.webp" alt="Project logo"></a>
</p>

<h3 align="center">My Convex Optimization</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/AlgorithmVoyager/MyOptimizationLib/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/AlgorithmVoyager/MyOptimizationLib/pulls)

</div>

---

<p align="center"> Hello, Guys. This is a Convex Optimization Project with C++.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

The Convex Optimization Project mains to deal with Convex Problem in C++ environment, it's my Algorithm & Code Practice Project.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

Before use the prooject, you might need to have a C++ enviroment, e.g. g++, gcc, clang, bazel

for g++, gcc:
sudo apt install xxx

for bazel:
https://bazel.build/install

### Installing

waiting...

## 🔧 Running the tests <a name = "tests"></a>

if you want to run all the tests with results:
bazel test --cxxopt=-std=c++17 --remote_cache="" --test_output=all //...

or if you just wanna run only one test:
bazel test --cxxopt=-std=c++17 --remote_cache="" --test_output=all //test/coretest/ConvexOptimizatoin/SteepestGradientDescent:steepest_gradient_descent_test

### Break down into end to end tests

### And coding style tests

## 🎈 Usage <a name="usage"></a>

if you wanna to use the lib, you can add the lib in your project. in Bazel, its very easy to do that, just append lib name in deps.

## 🚀 Deployment <a name = "deployment"></a>

I only test it on Ubuntu22.04, i think it may also works on other platforms

## ⛏️ Built Using <a name = "built_using"></a>

for build the project, i use gtest and glog, you can see them in WORKSPACE file. if it doesn't work for you, you can replace it as you local installed libs.

## ✍️ Authors <a name = "authors"></a>

- [@AlgorithmVoyager](https://github.com/AlgorithmVoyager)
- Idea & Initial work

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- References
  1. Steepest Gradient Descent(https://www.quora.com/Mathematical-Optimization-Why-does-the-method-of-steepest-descent-using-typical-gradient-descent-have-trouble-with-the-Rosenbrock-function)
