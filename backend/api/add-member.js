import jwt from 'jsonwebtoken';
import axios from 'axios';

export default async function handler(req, res) {
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
  const payload = { members: [{ email: req.body.email }] };

  try {
    const response = await axios.post(url, payload, { headers });
    res.status(200).json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.response.data.errors[0].message });
  }
}
