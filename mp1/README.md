# Machine Problem 1

## 1. Vector Space Model

### 1.1 Understand Zipf's Law

#### 1.1 - 1
Normalization implementation

```java
public String Normalization(String token) {
  // remove all non-word characters
  // please change this to removing all English punctuation

  // after tokenization, the actual punctuation should only appear in the either ends
  token = token.replaceAll("(^\\p{Punct}+|\\p{Punct}+$)", "");

  // convert to lower case
  token = token.toLowerCase();

  // add a line to recognize integers and doubles via regular expression

  // take care of formatted number with comma: 3,000.01
  Pattern numberPtn = Pattern.compile("\\d+(,\\d{3})*(\\.\\d+)?");
  Boolean isNumber = numberPtn.matcher(token).matches();

  // and convert the recognized integers and doubles to a special symbol "NUM"
  if (isNumber) {
    token = "NUM";
  }

  return token;
}
```

#### 1.1 - 2

The log-log scale plot of TTF

<img width="575" alt="TTF-plot" src="./TTF.png">

For the linear interpolation, the coefficient is `-1.43903409` and the intercept is `6.90280094`, which stands for `10^6.90280094` on y-axis.

#### 1.1 - 3

The log-log scale plot of DF

<img width="575" alt="DF-plot" src="./DF.png">

For the linear interpolation, the coefficient is `-1.41265775` and the intercept is `6.77386445`, which stands for `10^6.77386445` on y-axis.

TTF fits Zipf's law better than DF. The high-frequency words tend to occure multiple times in a long document. Document frequency makes these head works have exact the same weight with the words only appear once in a document. While the head words are heavily pubished, the tail words which rarely appear more than once in a document are not impacted. The above plots can also approve this. While both the linear interpolation can closely fit the center and tail parts of the dots, the head of DF has a much larger difference than TTF.
