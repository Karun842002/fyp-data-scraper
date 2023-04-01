import { useState } from 'react';
import './App.css';
import { CountUp } from 'use-count-up'
import axios from 'axios'
function App() {
  const [confidence, setConfidence] = useState([0, 50])
  const [input, setInput] = useState('')

  async function predict(event) {
    event.preventDefault()
    console.log(input)
    let res = await axios.get(`http://localhost:8000/predict?sentence=${input}`)
    console.log(res)
    setConfidence([0, res.data.confidence])
  }

  return (
    <div className="App">
      <div id="wrapper">
        <div className='header'>FactFinder</div>
        <div id="form-wrapper">
          <form onSubmit={predict}>
            <div className='form-element input-element'>
              <input type="text" id="input" placeholder="HEADLINE" name="sentence" onChange={event => setInput(event.target.value)} value={input} />
            </div>
            <div className='form-element counter' id="counter">
              TRUTH: <CountUp isCounting start={confidence[0]} end={confidence[1]} duration={2.5} decimalPlaces={4} updateInterval={0.1} onUpdate={(currentValue) => {
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
