# numpAI

**numpAI** is a lightweight implementation of neural networks using only NumPy. It provides a simple interface to build, train, and evaluate custom neural networks using common layer types.

⚠️ This project is still under development. Contributions and feedback are welcome!

## Features

- Modular architecture with a `Network` class and many `Layer` subclasses  
- Support for common layer types (`FCLayer`, `ActivationLayer`, etc.) 
- Forward and backward propagation implemented from scratch   
- Almost no external dependencies beyond NumPy  
- Easy to understand and use

## Installation

## Example Usage

## ⚠️ Security Warning

**Do NOT load a model file unless you trained it yourself or fully trust its source.**

The `load` method uses Python’s `pickle` module, which can execute arbitrary code during deserialization. This can pose serious security risks if the file has been tampered with.

For more information, see:  
- [Python Pickle Risks and Safer Alternatives](https://www.arjancodes.com/blog/python-pickle-module-security-risks-and-safer-alternatives/)  
- [Safely load a pickle file? – StackExchange](https://security.stackexchange.com/questions/183966/safely-load-a-pickle-file)

## See also
[This project](https://pypi.org/project/npnet/) is quite the same. But mine is better 😎.

## License

GNU AGPLv3 License. See `LICENSE` file for details.
