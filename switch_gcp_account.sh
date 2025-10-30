#!/bin/bash
# ============================================================================
# Switch to Different GCP Account and Redeploy
# ============================================================================

set -e

echo "========================================"
echo "GCP Account Switch & Redeploy"
echo "========================================"
echo ""

# Function to list accounts
list_accounts() {
    echo "Currently authenticated accounts:"
    gcloud auth list
    echo ""
}

# Function to add new account
add_new_account() {
    echo "To add a new Google Cloud account:"
    echo "Run: gcloud auth login"
    echo ""
    echo "This will open a browser window for authentication."
    echo "After authentication, the account will be added to gcloud."
    echo ""
    read -p "Would you like to add a new account now? (y/n): " add_account
    
    if [ "$add_account" = "y" ]; then
        gcloud auth login
        echo ""
        echo "✓ Account added successfully!"
        list_accounts
    fi
}

# Function to switch account
switch_account() {
    echo "Available accounts:"
    gcloud auth list
    echo ""
    read -p "Enter the email of the account to use: " account_email
    
    gcloud config set account "$account_email"
    echo ""
    echo "✓ Switched to account: $account_email"
}

# Function to list or create project
setup_project() {
    echo "========================================"
    echo "Project Setup"
    echo "========================================"
    echo ""
    
    echo "Listing projects for current account:"
    gcloud projects list 2>/dev/null || echo "No projects found or unable to list projects"
    echo ""
    
    read -p "Enter project ID to use (or 'create' to make new): " project_choice
    
    if [ "$project_choice" = "create" ]; then
        read -p "Enter new project ID (lowercase, hyphens allowed): " new_project_id
        read -p "Enter project name (human-readable): " project_name
        
        echo "Creating project: $new_project_id"
        gcloud projects create "$new_project_id" --name="$project_name"
        
        echo ""
        echo "✓ Project created!"
        PROJECT_ID="$new_project_id"
    else
        PROJECT_ID="$project_choice"
    fi
    
    echo ""
    echo "Setting project to: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
    
    echo ""
    echo "✓ Project configured!"
}

# Function to setup billing
setup_billing() {
    echo ""
    echo "========================================"
    echo "Billing Setup"
    echo "========================================"
    echo ""
    
    echo "Listing billing accounts:"
    gcloud billing accounts list
    echo ""
    
    read -p "Enter billing account ID (format: XXXXXX-XXXXXX-XXXXXX): " billing_id
    
    echo "Linking billing account to project..."
    gcloud billing projects link "$PROJECT_ID" --billing-account="$billing_id"
    
    echo ""
    echo "✓ Billing linked to project!"
}

# Function to deploy
deploy_instance() {
    echo ""
    echo "========================================"
    echo "Deploy Instance"
    echo "========================================"
    echo ""
    
    read -p "Enter Docker Hub username: " docker_user
    read -p "Enter deployment zone (e.g., asia-southeast1-b): " zone
    
    echo ""
    echo "Starting deployment..."
    echo "  Project: $PROJECT_ID"
    echo "  Zone: $zone"
    echo "  Docker user: $docker_user"
    echo ""
    
    # Run deployment script
    if [ -f "./deploy_to_gcp_cpu.sh" ]; then
        ./deploy_to_gcp_cpu.sh "$docker_user" "$PROJECT_ID" "$zone"
    else
        echo "❌ deploy_to_gcp_cpu.sh not found!"
        echo "Please ensure the deployment script exists in the current directory."
        exit 1
    fi
}

# Main execution
main() {
    echo "Step 1: Check current account"
    list_accounts
    echo ""
    
    read -p "Do you want to add a new account or switch to existing? (add/switch/keep): " choice
    
    case $choice in
        add)
            add_new_account
            ;;
        switch)
            switch_account
            ;;
        keep)
            echo "Keeping current account"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
    
    echo ""
    echo "Step 2: Setup project"
    setup_project
    
    echo ""
    read -p "Does this project have billing enabled? (y/n): " has_billing
    
    if [ "$has_billing" = "n" ]; then
        setup_billing
    else
        echo "✓ Billing already configured"
    fi
    
    echo ""
    echo "Step 3: Deploy instance"
    read -p "Ready to deploy? (y/n): " ready
    
    if [ "$ready" = "y" ]; then
        deploy_instance
    else
        echo "Deployment skipped."
        echo ""
        echo "To deploy later, run:"
        echo "./deploy_to_gcp_cpu.sh <docker_user> <project_id> <zone>"
    fi
    
    echo ""
    echo "========================================"
    echo "Configuration Complete!"
    echo "========================================"
    echo ""
    echo "Current configuration:"
    gcloud config list
}

# Run main function
main
