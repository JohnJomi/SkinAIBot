FROM node:20-slim

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# (Dev) Skip build; Vite dev server serves source with HMR.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
