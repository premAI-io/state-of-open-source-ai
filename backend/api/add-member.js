import jwt from 'jsonwebtoken';
import axios from 'axios';
import ZeroBounceSDK from '@zerobounce/zero-bounce-sdk';

export default async function handler(req, res) {
  // Set CORS headers
  const allowedOrigins = ['https://book.premai.io', 'http://localhost:8000'];
  const origin = req.headers.origin;
  if (allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Verify email address
  const zeroBounce = new ZeroBounceSDK();
  zeroBounce.init(process.env.ZEROBOUNCE_API_KEY);
  const email = req.body.email;

  try {
    const response = await zeroBounce.validateEmail(email);
    console.log("response", response);
    if (response.status === 'valid') {
      console.log(`Email ${email} is valid`);
    } else {
      console.error(response.status);
      return res.status(400).json({ error: `Invalid email address ${email}. Status: ${response.status}` });
    }
  } catch (error) {
    console.error("ZeroBounce error", error);
    return res.status(500).json({ error: error.message });
  }

  // Add member to Ghost
  const BLOG_URL = "https://prem.ghost.io";
  const [id, secret] = process.env.GHOST_ADMIN_API_KEY.split(':');
  const token = jwt.sign({}, Buffer.from(secret, 'hex'), {
    keyid: id,
    algorithm: 'HS256',
    expiresIn: '5m',
    audience: `/admin/`
  });

  const url = `${BLOG_URL}/ghost/api/admin/members/`;
  const headers = { Authorization: `Ghost ${token}` };
  const payload = {members: [{email, newsletters: [{id: "6575d0912c87960008d86bbd"}]}]};

  try {
    const response = await axios.post(url, payload, { headers });
    return res.status(200).json(response.data);
  } catch (error) {
    console.error("Ghost error", error.response.data.errors);
    if (error.response.data.errors[0].context === 'Member already exists. Attempting to add member with existing email address') {
      return res.status(200).json({});
    }
    return res.status(500).json({ error: `${error.response.data.errors[0].message} Context: ${error.response.data.errors[0].context}` });
  }
}
