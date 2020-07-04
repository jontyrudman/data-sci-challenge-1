# What have I done?

Mainly just been trying to understand whats going on with this task. I've made
a way of abstracting some of the complexity of operations within a wrapper
class.

# Why?

So for now there is no real benefit, but more functionality can be added down
the line.

# What is linReg.py?

Well you wanted something cool, so I delivered (hopefully). I decided to
write a program to optimise a shallow neural network for time and accuracy.

It takes absolutely ages to run (could probably be improved with a GPU) as it
does many cycles of training models.

I decided to use linear regression to predict what number of logits/neurons in
the hidden layer yield the most accurate/fast training.