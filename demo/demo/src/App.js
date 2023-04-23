import { useState } from 'react';
import './App.css';
import { CountUp } from 'use-count-up'
import axios from 'axios'
function App() {
  const [confidence, setConfidence] = useState(50)
  const [start, setStart] = useState(0)
  const [input, setInput] = useState('')

  function predict(event) {
    event.preventDefault()
    console.log(input)
    axios.get(`/predict?sentence=${input}`).then(res => {
      console.log(res)
      setStart(confidence)
      setConfidence(res.data.confidence)
    })
  }

  return (
    <div className="App">
      <div id="wrapper">
        <div className='header'>FactFinder</div>
        <div id="form-wrapper">
          <form onSubmit={predict}>
            <div className='form-element input-element'>
              <textarea type="text" id="input" placeholder="HEADLINE" name="sentence" onChange={event => setInput(event.target.value)} value={input} />
            </div>
            <div className='form-element counter' id="counter">
              TRUTH: <CountUp isCounting start={start} end={confidence} key={confidence} duration={2.5} decimalPlaces={4} updateInterval={0.1} onUpdate={(currentValue) => {
                document.getElementById("counter").style.color = `rgb(${255*Number(1 - currentValue/100)},${255*Number(currentValue/100)},${128 - 255*Math.abs(Number(0.5 - currentValue/100))})`
              }} />%
            </div>
            <div className='form-element'><button type="submit" className='button-85'>Predict</button></div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
