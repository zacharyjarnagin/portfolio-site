type Location = {
  city: string;
  state: string;
  country?: string;
}

type Stint = {
  title: string;
  start_date: string;
  end_date?: string;
  accomplishments: string[]
}

type Company = {
  name: string;
  iconUri?: string;
}

type Experience = {
  slug: string;
  company: Company;
  location: Location;
  stints: Stint[];
};

type ExperiencesResponse = {
  professional_experiences: Experience[];
};