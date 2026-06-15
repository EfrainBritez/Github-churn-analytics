-- ============================================
-- Schema: GitHub User Churn Analysis System
-- Database: PostgreSQL (Supabase)
-- Description: Stores GitHub user profiles, activity data, 
--              and engineered churn features
-- ============================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- 1. CORE USERS TABLE
-- Stores GitHub user profile data
-- ============================================
CREATE TABLE github_users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    github_id BIGINT UNIQUE NOT NULL,
    login VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    html_url TEXT,
    name VARCHAR(255),
    company VARCHAR(255),
    blog TEXT,
    location VARCHAR(255),
    email VARCHAR(255),
    hireable BOOLEAN,
    bio TEXT,
    twitter_username VARCHAR(255),
    public_repos INTEGER DEFAULT 0,
    public_gists INTEGER DEFAULT 0,
    followers INTEGER DEFAULT 0,
    following INTEGER DEFAULT 0,
    created_at_github TIMESTAMP WITH TIME ZONE,
    updated_at_github TIMESTAMP WITH TIME ZONE,
    -- Track when we first saw this user
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- User status in our system
    is_active BOOLEAN DEFAULT TRUE,
    churn_date TIMESTAMP WITH TIME ZONE,
    churn_predicted BOOLEAN DEFAULT FALSE
);

-- ============================================
-- 2. ORGANIZATIONS TABLE
-- Stores GitHub organization data
-- ============================================
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    github_id BIGINT UNIQUE NOT NULL,
    login VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    html_url TEXT,
    name VARCHAR(255),
    company VARCHAR(255),
    blog TEXT,
    location VARCHAR(255),
    email VARCHAR(255),
    description TEXT,
    public_repos INTEGER DEFAULT 0,
    public_gists INTEGER DEFAULT 0,
    followers INTEGER DEFAULT 0,
    following INTEGER DEFAULT 0,
    created_at_github TIMESTAMP WITH TIME ZONE,
    updated_at_github TIMESTAMP WITH TIME ZONE,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 3. USER-ORGANIZATION MEMBERSHIP
-- Tracks which users belong to which organizations
-- ============================================
CREATE TABLE user_organization_memberships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES github_users(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    role VARCHAR(50),
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, organization_id)
);

-- ============================================
-- 4. REPOSITORIES TABLE
-- Stores repository metadata and metrics
-- ============================================
CREATE TABLE repositories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    github_id BIGINT UNIQUE NOT NULL,
    owner_id UUID REFERENCES github_users(id),
    organization_id UUID REFERENCES organizations(id),
    name VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) UNIQUE NOT NULL,
    html_url TEXT,
    description TEXT,
    fork BOOLEAN DEFAULT FALSE,
    private BOOLEAN DEFAULT FALSE,
    language VARCHAR(100),
    license VARCHAR(100),
    topics TEXT[], -- Array of topics/tags
    stars_count INTEGER DEFAULT 0,
    forks_count INTEGER DEFAULT 0,
    watchers_count INTEGER DEFAULT 0,
    open_issues_count INTEGER DEFAULT 0,
    default_branch VARCHAR(255) DEFAULT 'main',
    created_at_github TIMESTAMP WITH TIME ZONE,
    updated_at_github TIMESTAMP WITH TIME ZONE,
    pushed_at_github TIMESTAMP WITH TIME ZONE,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Churn-related metadata
    last_activity_date TIMESTAMP WITH TIME ZONE,
    days_since_last_push INTEGER
);

-- ============================================
-- 5. REPOSITORY LANGUAGES
-- Stores language composition data
-- ============================================
CREATE TABLE repository_languages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    repository_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    language VARCHAR(100) NOT NULL,
    bytes_of_code BIGINT,
    percentage DECIMAL(5,2),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(repository_id, language)
);

-- ============================================
-- 6. COMMITS TABLE
-- Stores commit history
-- ============================================
CREATE TABLE commits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sha VARCHAR(40) UNIQUE NOT NULL,
    repository_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    author_github_id BIGINT,
    committer_github_id BIGINT,
    message TEXT,
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    total_changes INTEGER DEFAULT 0,
    files_changed INTEGER DEFAULT 0,
    committed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 7. ISSUES TABLE
-- Stores issue tracking data
-- ============================================
CREATE TABLE issues (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    github_id BIGINT UNIQUE NOT NULL,
    repository_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    author_id UUID REFERENCES github_users(id),
    assignee_id UUID REFERENCES github_users(id),
    title TEXT NOT NULL,
    body TEXT,
    state VARCHAR(20) DEFAULT 'open', -- open, closed
    labels TEXT[],
    milestone VARCHAR(255),
    comments_count INTEGER DEFAULT 0,
    created_at_github TIMESTAMP WITH TIME ZONE,
    updated_at_github TIMESTAMP WITH TIME ZONE,
    closed_at_github TIMESTAMP WITH TIME ZONE,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Churn signals
    days_to_close INTEGER,
    was_reopened BOOLEAN DEFAULT FALSE
);

-- ============================================
-- 8. PULL REQUESTS TABLE
-- Stores PR data and review metrics
-- ============================================
CREATE TABLE pull_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    github_id BIGINT UNIQUE NOT NULL,
    repository_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    author_id UUID REFERENCES github_users(id),
    title TEXT NOT NULL,
    body TEXT,
    state VARCHAR(20) DEFAULT 'open', -- open, closed, merged
    merged BOOLEAN DEFAULT FALSE,
    mergeable BOOLEAN,
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    changed_files INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    review_comments_count INTEGER DEFAULT 0,
    created_at_github TIMESTAMP WITH TIME ZONE,
    updated_at_github TIMESTAMP WITH TIME ZONE,
    merged_at_github TIMESTAMP WITH TIME ZONE,
    closed_at_github TIMESTAMP WITH TIME ZONE,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Review metrics
    reviewers_count INTEGER DEFAULT 0,
    time_to_first_review_hours DECIMAL(10,2),
    time_to_merge_hours DECIMAL(10,2),
    had_conflicts BOOLEAN DEFAULT FALSE
);

-- ============================================
-- 9. USER ACTIVITY EVENTS
-- Captures real-time user activity
-- ============================================
CREATE TABLE user_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    github_event_id BIGINT UNIQUE NOT NULL,
    user_id UUID NOT NULL REFERENCES github_users(id),
    event_type VARCHAR(100) NOT NULL, -- PushEvent, WatchEvent, CreateEvent, etc.
    repository_id UUID REFERENCES repositories(id),
    action VARCHAR(100),
    payload JSONB,
    created_at_github TIMESTAMP WITH TIME ZONE NOT NULL,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 10. USER SESSIONS (for our application)
-- Tracks when users interact with our platform
-- ============================================
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES github_users(id),
    session_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_end TIMESTAMP WITH TIME ZONE,
    duration_minutes INTEGER,
    actions_performed INTEGER DEFAULT 0,
    errors_encountered INTEGER DEFAULT 0,
    modules_accessed TEXT[],
    ip_address INET,
    user_agent TEXT
);

-- ============================================
-- 11. SUPPORT TICKETS
-- Tracks user support interactions
-- ============================================
CREATE TABLE support_tickets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES github_users(id),
    title TEXT NOT NULL,
    description TEXT,
    severity VARCHAR(20) DEFAULT 'normal', -- low, normal, high, critical
    status VARCHAR(20) DEFAULT 'open', -- open, in_progress, resolved, closed
    assigned_to VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_time_hours DECIMAL(10,2),
    reopened_count INTEGER DEFAULT 0,
    category VARCHAR(100)
);

-- ============================================
-- 12. SUBSCRIPTIONS / BILLING
-- Tracks user payment and plan information
-- ============================================
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES github_users(id),
    plan_name VARCHAR(100) NOT NULL,
    plan_tier VARCHAR(50), -- free, pro, enterprise
    monthly_amount DECIMAL(10,2) NOT NULL,
    start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_date TIMESTAMP WITH TIME ZONE,
    renewal_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active', -- active, cancelled, expired, trial
    payment_method VARCHAR(50),
    payment_status VARCHAR(20) DEFAULT 'paid', -- paid, pending, failed
    failed_payments_count INTEGER DEFAULT 0,
    last_payment_date TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE
);

-- ============================================
-- 13. CORE ACTIONS TRACKING
-- Records when users perform key value actions
-- ============================================
CREATE TABLE core_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES github_users(id),
    action_type VARCHAR(100) NOT NULL, -- e.g., 'export_report', 'run_analysis', 'deploy_code'
    action_name VARCHAR(255) NOT NULL,
    repository_id UUID REFERENCES repositories(id),
    metadata JSONB,
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 14. ENGINEERED FEATURES SNAPSHOT
-- Daily calculation of churn prediction features
-- This is the main table for ML model training
-- ============================================
CREATE TABLE user_features_snapshot (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES github_users(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
    
    -- 📊 Category 1: Engagement and Usage
    -- Feature 1: Velocity Ratio
    actions_last_7_days INTEGER DEFAULT 0,
    actions_last_30_days INTEGER DEFAULT 0,
    velocity_ratio DECIMAL(10,4),
    
    -- Feature 2: Feature Breadth (Depth of Adoption)
    distinct_modules_used INTEGER DEFAULT 0,
    total_core_modules_available INTEGER DEFAULT 5,
    feature_breadth DECIMAL(10,4),
    
    -- Feature 3: Core Action Intensity
    core_actions_count INTEGER DEFAULT 0,
    active_days_in_month INTEGER DEFAULT 0,
    core_action_intensity DECIMAL(10,4),
    
    -- ⚠️ Category 2: Friction and Technical Experience
    -- Feature 4: Error Rate Severity
    errors_last_14_days INTEGER DEFAULT 0,
    sessions_last_14_days INTEGER DEFAULT 0,
    error_rate_severity DECIMAL(10,4),
    
    -- Feature 5: Support Friction Time
    avg_ticket_resolution_hours DECIMAL(10,2),
    company_avg_resolution_hours DECIMAL(10,2) DEFAULT 24.00,
    support_friction_time DECIMAL(10,4),
    
    -- 💰 Category 3: Financial Behavior and Lifecycle
    -- Feature 6: Downgrade Momentum
    current_month_spend DECIMAL(10,2),
    avg_spend_last_3_months DECIMAL(10,2),
    downgrade_momentum DECIMAL(10,2),
    
    -- Feature 7: Renewal Danger Zone
    contract_renewal_date TIMESTAMP WITH TIME ZONE,
    days_until_renewal INTEGER,
    renewal_danger_zone INTEGER, -- days remaining
    
    -- Feature 8: Ghosting Rate (Post-Onboarding)
    days_since_registration INTEGER,
    days_until_first_value_action INTEGER,
    ghosting_rate INTEGER,
    
    -- Additional computed metrics
    overall_churn_probability DECIMAL(5,4),
    churn_risk_level VARCHAR(20), -- low, medium, high, critical
    model_version VARCHAR(50),
    
    -- Metadata
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    features_json JSONB, -- Store all features as JSON for flexibility
    
    UNIQUE(user_id, snapshot_date)
);

-- ============================================
-- 15. CHURN PREDICTIONS HISTORY
-- Stores model predictions over time
-- ============================================
CREATE TABLE churn_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES github_users(id) ON DELETE CASCADE,
    prediction_date DATE NOT NULL DEFAULT CURRENT_DATE,
    churn_probability DECIMAL(5,4),
    is_churn_predicted BOOLEAN DEFAULT FALSE,
    risk_level VARCHAR(20),
    top_contributing_features JSONB,
    model_version VARCHAR(50),
    model_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- INDEXES
-- Optimize query performance
-- ============================================

-- Users
CREATE INDEX idx_github_users_login ON github_users(login);
CREATE INDEX idx_github_users_company ON github_users(company);
CREATE INDEX idx_github_users_churn_date ON github_users(churn_date);
CREATE INDEX idx_github_users_active ON github_users(is_active);

-- Repositories
CREATE INDEX idx_repositories_owner ON repositories(owner_id);
CREATE INDEX idx_repositories_language ON repositories(language);
CREATE INDEX idx_repositories_stars ON repositories(stars_count DESC);
CREATE INDEX idx_repositories_last_push ON repositories(days_since_last_push);

-- Commits
CREATE INDEX idx_commits_repository ON commits(repository_id);
CREATE INDEX idx_commits_date ON commits(committed_at);
CREATE INDEX idx_commits_author ON commits(author_github_id);

-- Issues
CREATE INDEX idx_issues_repository ON issues(repository_id);
CREATE INDEX idx_issues_state ON issues(state);
CREATE INDEX idx_issues_closed_at ON issues(closed_at_github);

-- Pull Requests
CREATE INDEX idx_prs_repository ON pull_requests(repository_id);
CREATE INDEX idx_prs_state ON pull_requests(state);
CREATE INDEX idx_prs_merged ON pull_requests(merged);

-- User Events
CREATE INDEX idx_events_user ON user_events(user_id);
CREATE INDEX idx_events_type ON user_events(event_type);
CREATE INDEX idx_events_date ON user_events(created_at_github);

-- Features Snapshot
CREATE INDEX idx_features_user ON user_features_snapshot(user_id);
CREATE INDEX idx_features_date ON user_features_snapshot(snapshot_date);
CREATE INDEX idx_features_churn_risk ON user_features_snapshot(churn_risk_level);
CREATE INDEX idx_features_probability ON user_features_snapshot(overall_churn_probability);

-- Predictions
CREATE INDEX idx_predictions_user ON churn_predictions(user_id);
CREATE INDEX idx_predictions_date ON churn_predictions(prediction_date);
CREATE INDEX idx_predictions_risk ON churn_predictions(risk_level);

-- Subscriptions
CREATE INDEX idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_status ON subscriptions(status);
CREATE INDEX idx_subscriptions_renewal ON subscriptions(renewal_date);

-- Support Tickets
CREATE INDEX idx_tickets_user ON support_tickets(user_id);
CREATE INDEX idx_tickets_status ON support_tickets(status);
CREATE INDEX idx_tickets_severity ON support_tickets(severity);

-- ============================================
-- VIEWS FOR COMMON QUERIES
-- ============================================

-- View: Active users at risk of churn
CREATE VIEW users_at_risk AS
SELECT 
    gu.login,
    gu.name,
    gu.email,
    ufs.snapshot_date,
    ufs.velocity_ratio,
    ufs.error_rate_severity,
    ufs.downgrade_momentum,
    ufs.overall_churn_probability,
    ufs.churn_risk_level
FROM user_features_snapshot ufs
JOIN github_users gu ON ufs.user_id = gu.id
WHERE ufs.snapshot_date = CURRENT_DATE
  AND ufs.churn_risk_level IN ('high', 'critical')
  AND gu.is_active = TRUE
ORDER BY ufs.overall_churn_probability DESC;

-- View: Daily user engagement metrics
CREATE VIEW daily_engagement_metrics AS
SELECT 
    gu.login,
    ufs.snapshot_date,
    ufs.actions_last_7_days,
    ufs.actions_last_30_days,
    ufs.velocity_ratio,
    ufs.core_action_intensity,
    ufs.active_days_in_month,
    ufs.feature_breadth
FROM user_features_snapshot ufs
JOIN github_users gu ON ufs.user_id = gu.id
WHERE ufs.snapshot_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY ufs.snapshot_date DESC, ufs.velocity_ratio ASC;

-- View: Support ticket analysis
CREATE VIEW support_metrics AS
SELECT 
    gu.login,
    COUNT(st.id) as total_tickets,
    COUNT(CASE WHEN st.severity = 'critical' THEN 1 END) as critical_tickets,
    AVG(st.resolution_time_hours) as avg_resolution_time,
    ufs.support_friction_time
FROM support_tickets st
JOIN github_users gu ON st.user_id = gu.id
LEFT JOIN user_features_snapshot ufs ON ufs.user_id = gu.id 
    AND ufs.snapshot_date = CURRENT_DATE
WHERE st.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY gu.login, ufs.support_friction_time;

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function: Update last_updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for github_users
CREATE TRIGGER update_github_users_updated_at 
    BEFORE UPDATE ON github_users 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for repositories
CREATE TRIGGER update_repositories_updated_at 
    BEFORE UPDATE ON repositories 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Function: Calculate days since last activity
CREATE OR REPLACE FUNCTION calculate_days_since_last_push()
RETURNS TRIGGER AS $$
BEGIN
    NEW.days_since_last_push = EXTRACT(DAY FROM (NOW() - NEW.pushed_at_github));
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-calculate days since last push
CREATE TRIGGER calc_days_since_push 
    BEFORE INSERT OR UPDATE ON repositories 
    FOR EACH ROW 
    EXECUTE FUNCTION calculate_days_since_last_push();

-- ============================================
-- ROW LEVEL SECURITY (Supabase)
-- ============================================

-- Enable RLS on tables that need it
ALTER TABLE github_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_features_snapshot ENABLE ROW LEVEL SECURITY;
ALTER TABLE churn_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE support_tickets ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own data" ON github_users
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can view their own features" ON user_features_snapshot
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can view their own predictions" ON churn_predictions
    FOR SELECT USING (auth.uid() = user_id);

-- Admin policies (for your application backend)
CREATE POLICY "Service role full access" ON user_features_snapshot
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON churn_predictions
    FOR ALL USING (auth.role() = 'service_role');

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON TABLE user_features_snapshot IS 'Daily snapshot of engineered features for churn prediction. This is the main table for ML training data.';
COMMENT ON COLUMN user_features_snapshot.velocity_ratio IS 'Feature 1: Actions last 7 days / (Actions last 30 days / 4). Value < 0.5 indicates high churn risk.';
COMMENT ON COLUMN user_features_snapshot.feature_breadth IS 'Feature 2: Distinct modules used / Total core modules available. Low adoption increases churn risk.';
COMMENT ON COLUMN user_features_snapshot.core_action_intensity IS 'Feature 3: Core actions count / Active days in month. Measures value extraction frequency.';
COMMENT ON COLUMN user_features_snapshot.error_rate_severity IS 'Feature 4: Errors last 14 days / Sessions last 14 days. High values indicate frustration.';
COMMENT ON COLUMN user_features_snapshot.support_friction_time IS 'Feature 5: User avg resolution time / Company avg resolution time. Values > 2.0 are critical.';
COMMENT ON COLUMN user_features_snapshot.downgrade_momentum IS 'Feature 6: Current month spend - Avg spend last 3 months. Negative values indicate partial churn.';
COMMENT ON COLUMN user_features_snapshot.renewal_danger_zone IS 'Feature 7: Days until contract renewal. Low values with low engagement = high risk.';
COMMENT ON COLUMN user_features_snapshot.ghosting_rate IS 'Feature 8: Days since registration - Days until first value action. High values indicate failed onboarding.';