import jwt from 'jsonwebtoken';
import axios from 'axios';
import ZeroBounceSDK from '@zerobounce/zero-bounce-sdk';

export default async function handler(req, res) {
  // Verify email address
  const zeroBounce = new ZeroBounceSDK();
  zeroBounce.init(process.env.ZEROBOUNCE_API_KEY);
  const email = req.body.email;

  try {
    const response = await zeroBounce.validateEmail(email);
    if (response.status === 'Valid') {
      console.log(`Email ${email} is valid`);
    } else {
      console.error(response.status);
      res.status(400).json({ error: `Invalid email address ${email}. Status: ${response.status}` });
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }


  // Add member to Ghost
  const BLOG_URL = "https://prem.ghost.io";
  const [id, secret] = process.env.ADMIN_API_KEY.split(':');
  const token = jwt.sign({}, Buffer.from(secret, 'hex'), {
    keyid: id,
    algorithm: 'HS256',
    expiresIn: '5m',
    audience: `/admin/`
  });

  const url = `${BLOG_URL}/ghost/api/admin/members/`;
  const headers = { Authorization: `Ghost ${token}` };
  const payload = { members: [{ email }] };

  try {
    const response = await axios.post(url, payload, { headers });
    res.status(200).json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.response.data.errors[0].message });
  }
}
