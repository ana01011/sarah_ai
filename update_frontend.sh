#!/bin/bash

# Update Sarah AI Frontend from GitHub
# This script backs up current deployment and pulls latest from GitHub

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${GREEN}Starting Frontend Update Process...${NC}"

# Step 1: Create backup
echo -e "${YELLOW}Step 1: Creating backup...${NC}"
mkdir -p /root/frontend_backups
cd /root
tar -czf /root/frontend_backups/sarah_ai_backup_${TIMESTAMP}.tar.gz sarah_ai/
echo -e "${GREEN}✅ Backup saved to: /root/frontend_backups/sarah_ai_backup_${TIMESTAMP}.tar.gz${NC}"

# Step 2: Navigate to sarah_ai
echo -e "${YELLOW}Step 2: Updating from GitHub...${NC}"
cd /root/sarah_ai

# Save current git status
git status > /root/frontend_backups/git_status_${TIMESTAMP}.txt

# Method 1: Try to pull directly
echo -e "${YELLOW}Attempting to pull latest changes...${NC}"
git stash save "Auto-stash before update ${TIMESTAMP}"
git fetch origin
git reset --hard origin/main || git reset --hard origin/master
echo -e "${GREEN}✅ Latest code pulled from GitHub${NC}"

# Step 3: Install any new dependencies
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"
npm install
echo -e "${GREEN}✅ Dependencies updated${NC}"

# Step 4: Build for production
echo -e "${YELLOW}Step 4: Building for production...${NC}"
npm run build
echo -e "${GREEN}✅ Build completed${NC}"

# Step 5: Restart PM2
echo -e "${YELLOW}Step 5: Restarting PM2...${NC}"
pm2 restart sarah-frontend
pm2 save
echo -e "${GREEN}✅ PM2 restarted${NC}"

# Step 6: Verify
echo -e "${YELLOW}Step 6: Verifying deployment...${NC}"
sleep 3
if pm2 list | grep -q "sarah-frontend.*online"; then
    echo -e "${GREEN}✅ Frontend is running${NC}"
else
    echo -e "${RED}⚠️  Frontend might not be running properly${NC}"
fi

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Update completed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "📁 Backup location: /root/frontend_backups/sarah_ai_backup_${TIMESTAMP}.tar.gz"
echo "🌐 Access frontend at: http://147.93.102.165:3001"
echo ""
echo "Useful commands:"
echo "  pm2 logs sarah-frontend    - View logs"
echo "  pm2 status                 - Check status"
echo ""
echo -e "${YELLOW}Rollback command (if needed):${NC}"
echo "  tar -xzf /root/frontend_backups/sarah_ai_backup_${TIMESTAMP}.tar.gz -C /root/ && pm2 restart sarah-frontend"