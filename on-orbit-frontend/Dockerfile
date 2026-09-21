# Node 20 reached end of life; 22 is the current LTS line.
FROM node:22-slim

WORKDIR /app

# Copy manifests first so the dependency layer is cached across code changes.
COPY package.json package-lock.json ./

# `npm ci` installs exactly what package-lock.json pins and fails if the two
# have drifted, which is what a reproducible image wants. `npm install` would
# silently resolve newer versions and rewrite the lockfile inside the image.
RUN npm ci

COPY . .

EXPOSE 3000

# Development server, matching how docker-compose runs this stack. For
# production, `npm run build` and then `npm run start`.
CMD ["npm", "run", "dev"]
