#!/bin/bash

# Deploy Bolt Frontend Script
# Usage: ./deploy_bolt_frontend.sh <github-repo-url>

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if repo URL is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide the GitHub repository URL${NC}"
    echo "Usage: $0 <github-repo-url>"
    echo "Example: $0 https://github.com/username/repo.git"
    exit 1
fi

REPO_URL=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${GREEN}Starting Bolt Frontend Deployment...${NC}"

# Step 1: Backup current frontend
echo -e "${YELLOW}Step 1: Creating backup...${NC}"
cd /root
if [ -d "sarah_ai" ]; then
    tar -czf sarah_ai_backup_${TIMESTAMP}.tar.gz sarah_ai/
    echo -e "${GREEN}✓ Backup created: sarah_ai_backup_${TIMESTAMP}.tar.gz${NC}"
else
    echo -e "${YELLOW}No existing sarah_ai directory to backup${NC}"
fi

# Step 2: Clone new repository
echo -e "${YELLOW}Step 2: Cloning new repository...${NC}"
if [ -d "sarah_ai" ]; then
    mv sarah_ai sarah_ai_old_${TIMESTAMP}
fi
git clone $REPO_URL sarah_ai
cd sarah_ai
echo -e "${GREEN}✓ Repository cloned${NC}"

# Step 3: Install dependencies
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"
npm install
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Create environment file
echo -e "${YELLOW}Step 4: Configuring environment...${NC}"
cat > .env.production << 'EOF'
VITE_API_URL=http://147.93.102.165:8000
VITE_WS_URL=ws://147.93.102.165:8000
VITE_APP_NAME=Sarah AI
VITE_APP_VERSION=2.0.0
EOF
echo -e "${GREEN}✓ Environment configured${NC}"

# Step 5: Build for production
echo -e "${YELLOW}Step 5: Building for production...${NC}"
npm run build
echo -e "${GREEN}✓ Build completed${NC}"

# Step 6: Stop current PM2 process
echo -e "${YELLOW}Step 6: Stopping current frontend...${NC}"
pm2 stop sarah-frontend 2>/dev/null || true
echo -e "${GREEN}✓ Current frontend stopped${NC}"

# Step 7: Start with PM2
echo -e "${YELLOW}Step 7: Starting new frontend with PM2...${NC}"
pm2 delete sarah-frontend 2>/dev/null || true

# Create PM2 ecosystem file
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'sarah-frontend',
      script: 'npm',
      args: 'run preview',
      cwd: '/root/sarah_ai',
      env: {
        NODE_ENV: 'production',
        PORT: 3001,
        HOST: '0.0.0.0'
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G'
    }
  ]
};
EOF

pm2 start ecosystem.config.js
pm2 save
echo -e "${GREEN}✓ New frontend started${NC}"

# Step 8: Test deployment
echo -e "${YELLOW}Step 8: Testing deployment...${NC}"
sleep 5  # Wait for server to start
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3001 | grep -q "200\|304"; then
    echo -e "${GREEN}✓ Frontend is responding correctly${NC}"
else
    echo -e "${RED}⚠ Frontend may not be responding correctly${NC}"
    echo "Check logs with: pm2 logs sarah-frontend"
fi

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Access your frontend at: http://147.93.102.165:3001"
echo ""
echo "Useful commands:"
echo "  pm2 status          - Check status"
echo "  pm2 logs sarah-frontend - View logs"
echo "  pm2 restart sarah-frontend - Restart"
echo "  pm2 monit           - Monitor resources"
echo ""
echo -e "${YELLOW}Rollback command (if needed):${NC}"
echo "  cd /root && rm -rf sarah_ai && mv sarah_ai_old_${TIMESTAMP} sarah_ai && pm2 restart sarah-frontend"