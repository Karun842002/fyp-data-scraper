import { useState } from 'react';
import './App.css';
import CountUp from 'react-countup'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import axios from 'axios'
function App() {
  const [confidence, setConfidence] = useState([0,50])
  
  function predict() {
    setConfidence([confidence[1], 0])
    let headline = document.getElementById("input").nodeValue
    let res = axios.get(`http://localhost:5000/predict?sentence=${headline}`)
    setConfidence([0, res.confidence])
  }

  return (
    <div className="App">
      <div id="wrapper">
        <div className='header'>FactFinder <FontAwesomeIcon icon={'magnifying-glass'} /></div>
        <div id="form-wrapper">
        <form onSubmit={predict}>
          <div className='form-element input-element'><input type="text" id="input" placeholder="HEADLINE" name="sentence"></input></div>
          <div className='form-element counter'><CountUp start={confidence[0]} end={confidence[1]} decimals={4}></CountUp>%</div>
          <div className='form-element'><button type="submit" className='button-85'>Predict</button></div>
        </form>
        </div>
      </div>
    </div>
  );
}

export default App;
