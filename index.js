const express = require('express');
const crypto = require('crypto');
const {ethers} = require('ethers');
const app = express();
const path = require("path");
const bodyParser = require("body-parser");
app.use(express.static(__dirname));
const jwt = require('jsonwebtoken');
app.use(express.json())
app.use(bodyParser.json());

const PORT = 3000;

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname + '/views/index.html'));
});

app.get('/nounce', (req, res) => {
    const nounce = crypto.randomBytes(32).toString('hex');
    res.json({nounce});
});

const secretKey = 'mySecretKey';

app.post('/login', (req, res) => {
    console.log(req.body);
    const { signedMessage, message, address } = req.body;

    // Basic validation
    if (!signedMessage || !message || !address) {
        return res.status(400).json({ error: 'Missing parameters' });
    }

    try {
        // Verify the signed message
        const recoveredAddress = ethers.verifyMessage(message, signedMessage);
        console.log(`Recovered address: ${recoveredAddress}`);

        if (recoveredAddress.toLowerCase() !== address.toLowerCase()) {
            return res.status(401).json({ error: 'Invalid signature' });
        }

        // Generate the JWT token
        const token = jwt.sign({ address }, secretKey, { expiresIn: '1h' });
        console.log(`Generated token: ${token}`);

        // Send the JWT token to the frontend
        res.json({ token });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal server error' });
    }
});



// Endpoint for verifying the JWT token and logging in the user
app.post('/verify', (req, res) => {
    const authHeader = req.headers.authorization;
    
    const token = authHeader;
    
    try {
      // Verify the JWT token
      const decoded = jwt.verify(token, secretKey);
      console.log(decoded)
      const currentTime = Math.floor(Date.now() / 1000);
      console.log(currentTime)
      if (decoded.exp < currentTime) {
        res.json("tokenExpired");
      } else {
        res.json("ok");
      }

      
    } catch (err) {
      res.status(401).json({ error: 'Invalid token' });
    }
  });
  
  // Serve the success page
app.get('/success', (req, res) => {
    res.sendFile(path.join(__dirname + '/views/success.html'));
});


app.listen(PORT, ()=>{
    console.log(`Listening on port ${PORT}`); 
})